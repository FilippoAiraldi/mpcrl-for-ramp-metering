from typing import TYPE_CHECKING, TypeVar

import casadi as cs
import numpy as np
from csnlp.util.math import quad_form

from metanet import HighwayTrafficEnv
from util.constants import MpcRlConstants as MRC
from util.constants import a_, rho_crit_, v_free_

if TYPE_CHECKING:
    from mpc.highway_traffic_mpc import HighwayTrafficMpc

SymType = TypeVar("SymType", cs.SX, cs.MX)


def add_parametric_costs(
    mpc: "HighwayTrafficMpc[SymType]", env: HighwayTrafficEnv, gammas: cs.DM
) -> SymType:
    r"""Adds the parametric cost functions to the MPC controller. These are the
     - parametric initial cost: `V(s, weights)` (or \lambda_0)
     - paremtric stage cost: `L(s, rho_crit, v_free, weights)`
     - paremtric terminal cost: `T(s, rho_crit, v_free, weights)`

    Parameters
    ----------
    mpc : Mpc
        The MPC controller the costs will be added to.
    env : HighwayTrafficEnv
        The env the MPC controller is built for.
    gamma : cs.DM
        The discount factors.

    Returns
    -------
    cs.SX or MX
        The additional parametric costs to be added to the MPC objective.
    """
    # get variables from mpc
    s = mpc.states["s"]
    Np = s.shape[1] - 1
    n_segments, n_origins = env.n_segments, env.n_origins
    rho, v, w = cs.vertsplit(s, np.cumsum((0, n_segments, n_segments, n_origins)))

    # get normalization factors
    norm_rho, norm_v, norm_w = (
        MRC.normalization["rho"],
        MRC.normalization["v"],
        MRC.normalization["w"],
    )

    # add initial cost (affine in s[0])
    w_init_rho = mpc.parameter("weight_init_rho", (n_segments, 1))
    w_init_v = mpc.parameter("weight_init_v", (n_segments, 1))
    w_init_w = mpc.parameter("weight_init_w", (n_origins, 1))
    W = cs.vertcat(w_init_rho / norm_rho, w_init_v / norm_v, w_init_w / norm_w)
    J = cs.dot(W, s[:, 0])

    # add stage cost (quadratic in rho and v[1:-1])
    # NOTE: stage cost does not include first state s[0]
    w_stage_rho = mpc.parameter("weight_stage_rho", (n_segments, 1))
    w_stage_v = mpc.parameter("weight_stage_v", (n_segments, 1))
    w_stage_w = mpc.parameter("weight_stage_w", (n_origins, 1))
    rho_crit = MRC.wrong_dynamics["rho_crit"]
    v_free = MRC.wrong_dynamics["v_free"]
    for k in range(1, Np):
        J += gammas[k] * (
            quad_form(w_stage_rho, (rho[:, k] - rho_crit) / norm_rho)
            + quad_form(w_stage_v, (v[:, k] - v_free) / norm_v)
            + quad_form(w_stage_w, w[:, k] / norm_w)
        )

    # add terminal cost (quadratic in rho and v[-1])
    w_terminal_rho = mpc.parameter("weight_terminal_rho", (n_segments, 1))
    w_terminal_v = mpc.parameter("weight_terminal_v", (n_segments, 1))
    w_terminal_w = mpc.parameter("weight_terminal_w", (n_origins, 1))
    J += gammas[-1] * (
        quad_form(w_terminal_rho, (rho[:, -1] - rho_crit) / norm_rho)
        + quad_form(w_terminal_v, (v[:, -1] - v_free) / norm_v)
        + quad_form(w_terminal_w, w[:, -1] / norm_w)
    )

    # check parametric cost is scalar and return it
    assert J.is_scalar(), "Invalid parametric cost."
    return J
