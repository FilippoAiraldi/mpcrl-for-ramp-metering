from typing import TYPE_CHECKING, TypeVar

import casadi as cs
import numpy as np
from csnlp.util.math import quad_form

from metanet import HighwayTrafficEnv

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
    rho, v, _ = cs.vertsplit(s, np.cumsum((0, n_segments, n_segments, n_origins)))

    # add initial cost (affine in s)
    w_init_rho = mpc.parameter("weight_init_rho", (n_segments, 1))
    w_init_v = mpc.parameter("weight_init_v", (n_segments, 1))
    w_init_w = mpc.parameter("weight_init_w", (n_origins, 1))
    W = cs.vertcat(w_init_rho, w_init_v, w_init_w)
    J = cs.dot(W, s[:, 0])

    # add stage and terminal costs for speed (quadratic)
    v_free_stage = mpc.parameter("v_free_stage", (n_segments, 1))
    v_free_terminal = mpc.parameter("v_free_terminal", (n_segments, 1))
    w_stage_v = mpc.parameter("weight_stage_v", (n_segments, 1))
    w_terminal_v = mpc.parameter("weight_terminal_v", (n_segments, 1))
    W = cs.horzcat(
        *(gammas[k] * w_stage_v for k in range(Np)), gammas[-1] * w_terminal_v
    )
    V_FREE = cs.horzcat(cs.repmat(v_free_stage, 1, Np), v_free_terminal)
    J += quad_form(cs.vec(W), cs.vec(v - V_FREE))

    # add stage and terminal costs for speed (left-sided huber penalties as QPs)
    Z = mpc.variable("z", rho.shape)[0]  # auxiliary variable
    rho_crit_stage = mpc.parameter("rho_crit_stage", (n_segments, 1))
    rho_crit_terminal = mpc.parameter("rho_crit_terminal", (n_segments, 1))
    w_stage_rho_scale = mpc.parameter("weight_stage_rho_scale", (n_segments, 1))
    w_terminal_rho_scale = mpc.parameter("weight_terminal_rho_scale", (n_segments, 1))
    w_stage_rho_threshold = mpc.parameter("weight_stage_rho_threshold", (n_segments, 1))
    w_terminal_rho_threshold = mpc.parameter(
        "weight_terminal_rho_threshold", (n_segments, 1)
    )
    W_SCALE = cs.horzcat(
        *(gammas[k] * w_stage_rho_scale for k in range(Np)),
        gammas[-1] * w_terminal_rho_scale,
    )
    W_THRES = cs.horzcat(
        cs.repmat(w_stage_rho_threshold, 1, Np), w_terminal_rho_threshold
    )
    RHO_CRIT = cs.horzcat(cs.repmat(rho_crit_stage, 1, Np), rho_crit_terminal)
    J += quad_form(cs.vec(W_SCALE), cs.vec(Z)) - 2 * cs.dot(
        W_SCALE * W_THRES, rho - RHO_CRIT + Z
    )

    # add constraints on auxiliary variables
    mpc.constraint("Z_ineq_1", Z, "<=", W_THRES)
    mpc.constraint("Z_ineq_2", Z, "<=", RHO_CRIT - rho)

    # check parametric cost is scalar and return it
    assert J.is_scalar(), "Invalid parametric cost."
    return J
