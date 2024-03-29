from collections.abc import Iterable
from itertools import chain, takewhile
from typing import TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Nlp, Solution
from csnlp import multistart as ms
from csnlp.wrappers import Mpc
from sym_metanet import Network

from metanet import HighwayTrafficEnv
from mpc.costs import add_parametric_costs
from util.constants import EnvConstants as EC
from util.constants import MpcRlConstants as MRC

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HighwayTrafficMpc(Mpc[SymType]):
    """MPC controller for highway traffic control. This MPC formulation lends itself as
    function approximation for RL algorithms."""

    def __init__(
        self,
        env: HighwayTrafficEnv,
        discount: float,
        parametric_cost_terms: bool,
        seed: int | None = None,
    ) -> None:
        """Builds an instance of the MPC controller for traffic control.

        Parameters
        ----------
        env : HighwayTrafficEnv
            The traffic network the controller is acting upon.
        discount : float
            The discount factor for stage and terminal cost terms. Should be the same as
            the one used by the RL agent.
        parametric_cost_terms : bool
            If `True`, parametric initial, stage, and terminal cost terms are added to
            the MPC objective. If `False`, the objective is only made up of economic and
            traffic-related costs.
        seed : int, optional
            Optional seed for the random number generator. By default, `None`.
        """
        starts = MRC.structured_multistart + MRC.random_multistart + 1
        nlp = (
            Nlp(env.sym_type)
            if starts == 1
            else ms.ParallelMultistartNlp(env.sym_type, starts=starts, n_jobs=starts)
        )
        Np = MRC.prediction_horizon * EC.steps
        Nc = MRC.control_horizon * EC.steps
        super().__init__(nlp, Np, Nc, EC.steps)

        # create dynamics parameters
        pars = {n: self.parameter(n) for n in env.realpars}

        # create disturbances
        d = self.disturbance("d", env.nd)

        # create state variables
        s, _ = self.state("s", env.ns, lb=0)
        rho, _, w = cs.vertsplit(
            s, np.cumsum((0, env.n_segments, env.n_segments, env.n_origins))
        )

        # create action (flow of O2) and upper-constrain it dynamically
        # NOTE: infeasible problems may occur if demands are too low, due to the fact
        # that neither upper- nor lower-bounds are soft
        assert env.na == 1, "only 1 action is supported."
        i_ramp, i_seg = _find_index_of_ramp_and_segment(env.network, "O2")
        C = EC.origin_capacities[i_ramp]  # capacity of O2
        a, a_exp = self.action("a", env.na, lb=C / EC.ramp_min_flow_factor, ub=C)
        self.constraint("a_min_1", a_exp, "<=", d[i_ramp, :] + w[i_ramp, :-1] / EC.T)
        self.constraint(
            "a_min_2",
            (EC.rho_max - pars["rho_crit"]) * a_exp,
            "<=",
            C * (EC.rho_max - rho[i_seg, :-1]),
        )

        # create (soft) constraints on queue(s)
        for oi, origin in enumerate(env.network.origins_by_name):
            w_max = EC.ramp_max_queue.get(origin, None)
            if w_max is not None:
                self.constraint(f"w_max_{origin}", w[oi, :], "<=", w_max, soft=True)
        slacks = cs.vertcat(*self.slacks.values())

        # set dynamics
        p = cs.vertcat(*pars.values())
        F = lambda x, u, d: env.dynamics(x, u, d, p)[0]
        self.set_dynamics(F, n_in=3, n_out=1)

        # build objective terms related to traffic, control action, and slacks
        J = self.nlp.sym_type.zeros(1, 1)
        gammas = cs.DM(discount ** np.arange(Np + 1).reshape(1, -1))

        # total-time spent
        weight_tts = self.parameter("weight_tts")
        J += weight_tts * cs.dot(gammas, env.stage_cost(s, 0, 0)[0])

        # control action variability
        a_last = self.parameter("a-", (env.na, 1))
        a_lasts = cs.horzcat(a_last, a[:, :-1])
        weight_var = self.parameter("weight_var")
        J += weight_var * cs.dot(
            gammas[: Nc : EC.steps], env.stage_cost(0, a, a_lasts)[1]
        )

        # slack penalty
        weight_slack = self.parameter("weight_slack", slacks.shape)
        J += cs.dot(gammas * weight_slack, slacks)

        # add the additional parametric costs
        if parametric_cost_terms:
            J += add_parametric_costs(self, env, gammas)

        # set the MPC objective
        self.minimize(cs.simplify(J))

        # initialize solver
        self.init_solver(MRC.solver_opts)

        # initialize multistart point generators (only if multistart is on)
        self.structured_start_points = self.random_start_points = None
        if nlp.is_multi:
            bounds_and_size = {
                "s": (1e-3, 200, s.shape),
                "a": (C / EC.ramp_min_flow_factor, C, a.shape),
            }
            if MRC.structured_multistart > 0:
                self.structured_start_points = ms.StructuredStartPoints(
                    {
                        n: ms.StructuredStartPoint(lb, ub)
                        for n, (lb, ub, _) in bounds_and_size.items()
                    },
                    MRC.structured_multistart,
                )
            if MRC.random_multistart > 0:
                self.random_start_points = ms.RandomStartPoints(
                    {
                        n: ms.RandomStartPoint("uniform", *args)
                        for n, args in bounds_and_size.items()
                    },
                    MRC.random_multistart,
                    seed,
                )

    def __call__(
        self,
        pars: dict[str, npt.ArrayLike] | None = None,
        vals0: dict[str, npt.ArrayLike] | None = None,
    ) -> Solution[SymType] | list[Solution[SymType]]:
        if not self.nlp.is_multi:
            return self.nlp.solve(pars, vals0)

        vals0_: Iterable[dict[str, npt.ArrayLike]] = ()
        if self.structured_start_points is not None:
            vals0_ = chain(vals0_, iter(self.structured_start_points))
        if self.random_start_points is not None:
            vals0_ = chain(vals0_, iter(self.random_start_points))
        if vals0 is not None:
            vals0_ = chain(vals0_, (vals0,))
        return self.nlp.solve_multi(pars, vals0_)


def _find_index_of_ramp_and_segment(network: Network, rampname: str) -> tuple[int, int]:
    """Internal utility to find the index of a ramp and associated segment."""
    ramp = network.origins_by_name[rampname]
    link_with_ramp = next(iter(network.out_links(network.origins[ramp])))[2]
    idx_ramp = list(network.origins).index(ramp)  # index of O2
    idx_seg = sum(
        link[2].N
        for link in takewhile(lambda l: l[2] is not link_with_ramp, network.links)
    )
    return idx_ramp, idx_seg
