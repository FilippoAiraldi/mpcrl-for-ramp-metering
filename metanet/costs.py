import casadi as cs
from sym_metanet import Network


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
    L = cs.Function("L", [s, a, a_prev], [TTS, VAR], ["s", "a", "a-"], ["tts", "var"])
    return L
