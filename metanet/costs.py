from typing import TypeVar

import casadi as cs
import numpy as np
from sym_metanet import MeteredOnRamp, Network, Origin
from sym_metanet import engine as sm_engine

from util import EnvConstants as EC

SymType = TypeVar("SymType", cs.SX, cs.MX)


def get_stage_cost(
    network: Network,
    n_actions: int,
    T: float,
    w_max: dict[Origin, int],
    ramp_with_erm: MeteredOnRamp,
) -> cs.Function:
    """Returns the stage cost function to compute for each state-action pair the
    corresponding cost.

    Parameters
    ----------
    network : sym_metanet.Network
        The network in use.
    n_actions : int
        Number of actions available in the network.
    T : float
        Simulation timestep.
    w_max : dict[sym_metanet.Origins, int]
        A dictionary of origins and their corresponding threshold on the queue size.
    ramp_with_erm : sym_metanet.MeteredOnRamp
        Ramp in which effective ramp metering rate will be rewarded.

    Returns
    -------
    casadi.Function
        A CasADi function for the stage cost in the form `L(s,a,a-,q)`, where `s` is the
        network's state, `a` the action, `a-` the action at the previous time step, and
        `q` the flows in the network. The function returns the TTS cost, the control
        VARiability cost, the Constraint VIolation cost, and the Effective Ramp Metering
        reward as `TTS, VAR, CVI, ERM = L(s,a,a-,q).`

    Raises
    ------
    ValueError
        Raises if an origin name cannot be found in the given network.
    """
    symvar = cs.MX  # faster function evaluations with MX
    links = network.nodes_by_link.keys()
    origins = network.origins

    # compute Total-Time-Spent for the current state
    TTS = symvar.zeros(1, 1)
    rhos, vs = [], []
    for link in links:
        rho = symvar.sym(f"rho_{link.name}", link.N, 1)
        v = symvar.sym(f"v_{link.name}", link.N, 1)
        rhos.append(rho)
        vs.append(v)
        TTS += cs.sum1(rho) * link.lam * link.L
    ws = []
    for origin in origins:
        w = symvar.sym(f"w_{origin.name}", 1, 1)
        ws.append(w)
        TTS += cs.sum1(w)
    TTS *= T

    # compute control input variability
    a = symvar.sym("a", n_actions, 1)
    a_prev = symvar.sym("a_prev", n_actions, 1)
    VAR = cs.sumsqr(a - a_prev)

    # compute constraint violations for origins
    CVI = cs.vertcat(
        *(w - w_max[origin] for origin, w in zip(origins, ws) if origin in w_max)
    )

    # compute effective ramp metering
    upstream_links = list(network.in_links(origins[ramp_with_erm]))
    assert len(upstream_links) == 1, "Multiple upstream links for ramp with ERM."
    link = upstream_links[0][2]
    link_idx = list(links).index(link)
    origin_idx = list(origins).index(ramp_with_erm)
    d_ramp = symvar.sym(f"d_{ramp_with_erm.name}")
    ERM = _get_effective_ramp_metering(
        d_ramp,
        ws[origin_idx],
        sm_engine.links.get_flow(rhos[link_idx][-1], vs[link_idx][-1], link.lam),
        ramp_with_erm.C / EC.ramp_min_flow_factor,
        ramp_with_erm.C,
        2000 * link.N,
    )

    # pack into function L(s,a) (with a third argument for the previous action)
    assert (
        TTS.shape == (1, 1) and VAR.shape == (1, 1) and ERM.shape == (1, 1)
    ), "Non-scalar costs."
    s = cs.vertcat(*rhos, *vs, *ws)
    return cs.Function(
        "L",
        (s, a, a_prev, d_ramp),
        (TTS, VAR, CVI, ERM),
        ("s", "a", "a-", d_ramp.name()),
        ("tts", "var", "cvi", "erm"),
    )


def _get_effective_ramp_metering(
    d_ramp: SymType,
    w_ramp: SymType,
    q_link: SymType,
    ramp_min_flow: int,
    ramp_max_flow: int,
    link_capacity: int,
) -> SymType:
    """Internal utility to compute the ERM boolean for the current ramp and link."""
    # find the three points defining the triangle containg all flows for which ramp
    # metering is effective (areas X and W in pag. 87, A. Hegyi's PhD Thesis)
    q_ramp = d_ramp + w_ramp / EC.T
    q_drop = 0.9 * link_capacity
    points = np.asarray(
        (
            (ramp_min_flow, q_drop - ramp_min_flow),
            (ramp_max_flow, q_drop - ramp_min_flow),
            (ramp_max_flow, q_drop - ramp_max_flow),
        )
    )
    C = points.mean(0)  # triangle's centroid
    points = (1 - EC.erm_robustness + 1e-6) * (points - C) + C  # shrink triangle

    # create a function to check if a new point is in the inside of the triangle
    # https://mathworld.wolfram.com/TriangleInterior.html
    v = cs.vertcat(q_ramp, q_link)  # query point
    v0 = points[0]
    v1 = points[1] - v0
    v2 = points[2] - v0
    det_vv1 = _det(v, v1)
    det_vv2 = _det(v, v2)
    det_v0v1 = _det(v0, v1)
    det_v0v2 = _det(v0, v2)
    det_v1v2 = _det(v1, v2)
    a = (det_vv2 - det_v0v2) / det_v1v2
    b = (det_v0v1 - det_vv1) / det_v1v2
    return cs.logic_and(cs.logic_and(a >= 0, b >= 0), a + b <= 1)


def _det(u, v):
    """Internal utility to compute the determinant of [u, v]."""
    return u[0] * v[1] - u[1] * v[0]
