from typing import TypeVar

import casadi as cs
import numpy as np
from csnlp import MultistartNlp, Nlp
from csnlp.wrappers import Mpc

from metanet.highway_traffic_env import HighwayTrafficEnv
from util.constants import EnvConstants as EC
from util.constants import MpcConstants as MC

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HighwayTrafficMpc(Mpc[SymType]):
    __slots__ = ("env",)

    def __init__(self, env: HighwayTrafficEnv) -> None:
        self.env = env
        starts = MC.multistart
        nlp = (
            Nlp(sym_type=env.sym_type)
            if starts == 1
            else MultistartNlp(starts=starts, sym_type=env.sym_type)
        )
        sp = MC.input_spacing
        Np = MC.prediction_horizon * sp
        Nc = MC.control_horizon * sp
        super().__init__(nlp, Np, Nc, sp)

        # create parameters
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
        _, a_exp = self.action("a", env.na, ub=C)  # control action of O2
        self.constraint("a_min_1", a_exp, "<=", d[si, :] + w[si, :-1] / EC.T)
        self.constraint(
            "a_min_2",
            (EC.rho_max - pars["rho_crit"]) * a_exp,
            "<=",
            C * (EC.rho_max - rho[si, :-1]),
        )

        # create (soft) constraints on queue(s)
        # NOTE: no constraint on w[:, 0], since this must be equal to the init condition
        for oi, origin in enumerate(env.network.origins_by_name):
            if origin in EC.w_max:
                self.constraint(
                    f"w_max_O{oi + 1}", w[oi, 1:], "<=", EC.w_max[origin], soft=True
                )

        # set dynamics
        p = cs.vertcat(*pars.values())
        F = lambda x, u, d: env.dynamics(x, u, d, p)
        self.set_dynamics(F, n_in=3, n_out=env.dynamics.n_out())
