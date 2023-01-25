from typing import Callable, Literal, TypeVar, Union
from warnings import warn

import casadi as cs
import numpy as np
import sym_metanet as metanet

Tx = TypeVar("Tx")


def get_network(
    segment_length: float,
    lanes: int,
    origin_capacities: tuple[float, float],
    rho_max: float,
    sym_type: Literal["SX", "MX"],
) -> tuple[metanet.Network, dict[str, Union[cs.SX, cs.MX]]]:
    """Builds the target highway network.

    Parameters
    ----------
    segment_length : float
        Length of each segment in L1 (2-segment link) and L2 (1-segment link)
    lanes : int
        Number of lanes in L1 and L2
    origin_capacities : tuple[float, float]
        Capacity of main origin O1 and on-ramp O2.
    rho_max : float
        Maximum density
    sym_type : 'SX' or 'MX'
        Type of CasADi symbolic variable.

    Returns
    -------
    sym_metanet.Network
        The network instance describing the highway stretch.
    dict of str-symvar
        The symbolic variables for `rho_crit`, `a`, and `v_free`.
    """
    metanet.engines.use("casadi", sym_type=sym_type)
    a_sym = metanet.engine.var("a")  # model parameter (adim)
    v_free_sym = metanet.engine.var("v_free")  # free flow speed (km/h)
    rho_crit_sym = metanet.engine.var("rho_crit_sym")  # critical capacity (veh/km/lane)
    N1 = metanet.Node(name="N1")
    N2 = metanet.Node(name="N2")
    N3 = metanet.Node(name="N3")
    L1 = metanet.Link(
        2, lanes, segment_length, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L1"
    )
    L2 = metanet.Link(
        1, lanes, segment_length, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L2"
    )
    O1 = metanet.MeteredOnRamp(origin_capacities[0], name="O1")
    O2 = metanet.MeteredOnRamp(origin_capacities[1], "out", name="O2")
    D1 = metanet.CongestedDestination(name="D1")
    net = (
        metanet.Network()
        .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
        .add_origin(O2, N2)
    )
    net.is_valid(raises=True)
    return net, {"rho_crit": rho_crit_sym, "a": a_sym, "v_free": v_free_sym}


def steady_state(
    f: Callable[[Tx], Tx],
    x0: Tx,
    tol: float = 1e-3,
    maxiter: int = 500,
) -> tuple[Tx, float, int]:
    r"""Searches the steady-state of the given dynamics.

    Parameters
    ----------
    f : Callable[[Tx], Tx]
        The dynamics function of the form `x+ = f(x)`, where `x` is the state and `x+`
        the state at the next time instant.
    x0 : Tx
        The initial steady-state guess. The type must be compatible with numpy.
    tol : float, optional
        Error tolerance for convergence, by default 1e-3.
    maxiter : int, optional
        Maximum iterations, by default 500.

    Returns
    -------
    Tx
        The steady-state.
    error
        The residual error (less than tolerance, unless error is increasing or maximum
        iteration treshold has been reached)
    iters
        The actual number of iterations performed by the algorithm.
    """
    err_previous = float("inf")
    for k in range(maxiter):
        x0_ss = f(x0)
        err = float(np.linalg.norm(x0_ss - x0))  # type: ignore[operator]
        if err < tol:
            return x0_ss, err, k
        elif err > err_previous:
            warn(
                "Increasing error encountered in steady-state search "
                f"({err} > {err_previous})."
            )
            return x0_ss, err, k

        x0 = x0_ss
        err_previous = err

    warn(f"Maximum number of iterations reached in steady-state search ({maxiter}).")
    return x0_ss, err, k
