from typing import Dict

import casadi as cs
from sym_metanet import Network, Origin


def get_stage_cost(
    network: Network,
    n_actions: int,
    T: float,
    w_max: Dict[Origin, int],
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
    w_max : Dict[sym_metanet.Origins, int]
        A dictionary of origins and their corresponding threshold on the queue size.

    Returns
    -------
    casadi.Function
        A CasADi function for the stage cost in the form `L(s,a,a-)`, where `s` is the
        network's state, `a` the action, and `a-` the action at the previous time step.
        The function returns the TTS cost, the control VARiability cost, and the
        Constraint VIolation cost as `TTS, VAR, CVI = L(s,a,a-).`

    Raises
    ------
    ValueError
        Raises if an origin name cannot be found in the given network.
    """
    # compute Total-Time-Spent for the current state
    TTS = 0
    rhos, vs = [], []
    for _, _, link in network.links:
        rho = cs.MX.sym(f"rho_{link.name}", link.N, 1)
        v = cs.MX.sym(f"v_{link.name}", link.N, 1)
        rhos.append(rho)
        vs.append(v)
        TTS += cs.sum1(rho) * link.lam * link.L
    ws = []
    for origin in network.origins:
        w = cs.MX.sym(f"w_{origin.name}", 1, 1)
        ws.append(w)
        TTS += cs.sum1(w)
    TTS *= T  # type: ignore[assignment]

    # compute control input variability
    a = cs.MX.sym("a", n_actions, 1)
    a_prev = cs.MX.sym("a_prev", n_actions, 1)
    VAR = cs.sumsqr(a - a_prev)

    # compute constraint violations for origins
    CVI = cs.vertcat(
        *(
            w - w_max[origin]
            for origin, w in zip(network.origins, ws)
            if origin in w_max
        )
    )

    # pack into function L(s,a) (with a third argument for the previous action)
    s = cs.vertcat(*rhos, *vs, *ws)
    return cs.Function(
        "L", [s, a, a_prev], [TTS, VAR, CVI], ["s", "a", "a-"], ["tts", "var", "cvi"]
    )
