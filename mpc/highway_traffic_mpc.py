from typing import TypeVar

import casadi as cs
import numpy as np
from csnlp import StackedMultistartNlp, Nlp
from csnlp.wrappers import Mpc

from metanet.highway_traffic_env import HighwayTrafficEnv
from mpc.costs import get_parametric_cost
from util.constants import EnvConstants as EC
from util.constants import MpcConstants as MC

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HighwayTrafficMpc(Mpc[SymType]):
    """MPC controller for highway traffic control. This MPC formulation lends itself as
    function approximation for RL algorithms."""

    def __init__(
        self,
        env: HighwayTrafficEnv,
        discount: float,
        parametric_cost_terms: bool,
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
        """
        starts = MC.multistart
        nlp = (
            Nlp(env.sym_type)
            if starts == 1
            else StackedMultistartNlp(env.sym_type, starts=starts)
        )
        Np = MC.prediction_horizon * EC.steps
        Nc = MC.control_horizon * EC.steps
        super().__init__(nlp, Np, Nc, EC.steps)

        # create dynamics parameters
        pars = {n: self.parameter(n) for n in env.realpars.keys()}

        # create disturbances
        d = self.disturbance("d", env.nd)

        # create state variables
        s, _ = self.state("s", env.ns, lb=0)
        n_segments, n_origins = env.n_segments, env.n_origins
        rho, _, w = cs.vertsplit(s, np.cumsum((0, n_segments, n_segments, n_origins)))

        # create action and upper-constrain it dynamically
        C = EC.origin_capacities[1]  # capacity of O2
        si = 1  # index of segment connected to O2
        a, a_exp = self.action("a", env.na, lb=0, ub=C)  # control action of O2
        self.constraint("a_min_1", a_exp, "<=", d[si, :] + w[si, :-1] / EC.T)
        self.constraint(
            "a_min_2",
            (EC.rho_max - pars["rho_crit"]) * a_exp,
            "<=",
            C * (EC.rho_max - rho[si, :-1]),
        )

        # create (soft) constraints on queue(s)
        # NOTE: no constraint on w[:, 0], since this must be equal to init conditions.
        # For this reason, we prepend a column of zeros to make it equal size to state
        for oi, origin in enumerate(env.network.origins_by_name):
            if origin in EC.w_max:
                self.constraint(
                    f"w_max_O{oi + 1}", w[oi, 1:], "<=", EC.w_max[origin], soft=True
                )
        slacks = cs.horzcat(
            self.nlp.sym_type.zeros(self.nslacks, 1), cs.vertcat(*self.slacks.values())
        )

        # set dynamics
        p = cs.vertcat(*pars.values())
        F = lambda x, u, d: env.dynamics(x, u, d, p)[0]
        self.set_dynamics(F, n_in=3, n_out=1)

        # build objective terms related to traffic, control action, and slacks
        gammas = cs.DM(discount ** np.arange(Np + 1).reshape(1, -1))
        # total-time spent
        J = cs.dot(gammas, env.stage_cost(s, 0, 0)[0])
        # control action variability
        a_last = self.parameter("a-", (env.na, 1))
        a_lasts = cs.horzcat(a_last, a[:, :-1])
        weight_var = self.parameter("weight_var")
        J += weight_var * cs.sum2(env.stage_cost(0, a, a_lasts)[1])
        # slack penalty
        weight_slack = self.parameter("weight_slack", (slacks.shape[0], 1))
        weight_slack_T = self.parameter("weight_slack_terminal", (slacks.shape[0], 1))
        J += cs.dot(gammas[:-1], weight_slack.T @ slacks[:, :-1]) + gammas[-1] * cs.dot(
            weight_slack_T, slacks[:, -1]
        )

        # add the additional parametric costs
        # NOTE: should terminal cost have separate tracking values from stage cost?
        if parametric_cost_terms:
            init_cost, stage_cost, terminal_cost = get_parametric_cost(env.network)
            weight_V = self.parameter("weight_V", init_cost.size_in("weight"))
            weight_L = self.parameter("weight_L", stage_cost.size_in("weight"))
            weight_T = self.parameter("weight_T", terminal_cost.size_in("weight"))
            rho_crit_L = self.parameter("rho_crit_stage")
            v_free_L = self.parameter("v_free_stage")
            rho_crit_T = self.parameter("rho_crit_terminal")
            v_free_T = self.parameter("v_free_terminal")
            J += (
                init_cost(s[:, 0], weight_V)
                + cs.dot(
                    gammas[1:-1], stage_cost(s[:, 1:-1], rho_crit_L, v_free_L, weight_L)
                )
                + gammas[-1] * terminal_cost(s[:, -1], rho_crit_T, v_free_T, weight_T)
            )

        # set the MPC objective
        self.minimize(J)

        # initialize solver
        self.init_solver(MC.solver_opts)

    # TODO: override call to create multiple vals0 with some noise if multistart > 1
    # def __call__(self, *args, **kwargs):
    #   if multistart == 1
    #       call normal mpc
    #   else
    #       create multiple different vals0
    #       call multistart mpc
