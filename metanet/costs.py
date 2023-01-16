from typing import Dict

import casadi as cs
from sym_metanet import Network, Origin


def get_stage_cost(network: Network, n_actions: int, T: float) -> cs.Function:
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

    Returns
    -------
    casadi.Function
        A CasADi function for the stage cost in the form `L(s,a,a-)`, where `s` is the
        network's state, `a` the action, and `a-` the action at the previous time step.
    """
    # compute Total-Time-Spent for the current state
    TTS = 0
    rhos, vs = [], []
    for _, _, link in network.links:
        rho = cs.SX.sym(f"rho_{link.name}", link.N, 1)
        v = cs.SX.sym(f"v_{link.name}", link.N, 1)
        rhos.append(rho)
        vs.append(v)
        TTS += cs.sum1(rho) * link.lam * link.L
    ws = []
    for origin in network.origins:
        w = cs.SX.sym(f"w_{origin.name}", 1, 1)
        ws.append(w)
        TTS += cs.sum1(w)
    TTS *= T  # type: ignore[assignment]

    # compute control input variability
    a = cs.SX.sym("a", n_actions, 1)
    a_prev = cs.SX.sym("a_prev", n_actions, 1)
    VAR = cs.sumsqr(a - a_prev)

    # pack into function L(s,a) (with a third argument for the previous action)
    s = cs.vertcat(*rhos, *vs, *ws)
    return cs.Function(
        "L", [s, a, a_prev], [TTS, VAR], ["s", "a", "a-"], ["tts", "var"]
    )


def get_constraint_violation(
    network: Network, w_max: Dict[Origin, int], nonnegative: bool = False
) -> cs.Function:
    """Returns the function that evaluates the constraint violation for the current
    state.

    Parameters
    ----------
    network : sym_metanet.Network
        The network in use.
    w_max : Dict[sym_metanet.Origin, int]
        A dictionary of origins and their corresponding threshold on the queue size.
    nonnegative : bool, optional
        Allows for constraints that can only be positive or zero, by default `False`.

    Returns
    -------
    cs.Function
        A CasADi function of the form `CVI(s)` with returns the constraint violation for
        the state `s`. If `nonnegative=True`, computes `max(0, CVI(s))` instead.
    """
    # build symbolic variables
    n_segments = sum(link.N for _, _, link in network.links)
    n_origins = len(network.origins)
    rho = cs.SX.sym("rho", n_segments, 1)
    v = cs.SX.sym("v", n_segments, 1)
    w = cs.SX.sym("w", n_origins, 1)

    # compute constraints symbolically
    constraints = cs.vertcat(
        *(
            w[i] - w_max[origin]
            for i, origin in enumerate(network.origins)
            if origin in w_max
        )
    )
    if nonnegative:
        constraints = cs.fmax(0, constraints)

    # pack into function CVI(s)
    s = cs.vertcat(rho, v, w)
    return cs.Function("CVI", [s], [constraints], ["s"], ["cvi"])
