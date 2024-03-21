from collections.abc import Callable
from typing import Literal, TypeVar
from warnings import warn

import casadi as cs
import numpy as np
import sym_metanet as mn

Tx = TypeVar("Tx")


def get_network(
    segment_length: float,
    lanes: int,
    origin_capacities: tuple[float, float],
    rho_max: float,
    sym_type: Literal["SX", "MX"],
    control_O2_rate: bool = False,
) -> tuple[mn.Network, dict[str, cs.SX | cs.MX]]:
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
    control_O2_rate : bool, optional
        A flag to indicate whether the on-ramp O2 should be controlled by rate or by
        flow; by default `False`.

    Returns
    -------
    sym_metanet.Network
        The network instance describing the highway stretch.
    dict of str-symvar
        The symbolic variables for `rho_crit`, `a`, and `v_free`.
    """
    mn.engines.use("casadi", sym_type=sym_type)
    a_sym = mn.engine.var("a")  # model parameter (adim)
    v_free_sym = mn.engine.var("v_free")  # free flow speed (km/h)
    rho_crit_sym = mn.engine.var("rho_crit_sym")  # critical capacity (veh/km/lane)
    N1 = mn.Node(name="N1")
    N2 = mn.Node(name="N2")
    N3 = mn.Node(name="N3")
    L1 = mn.Link(
        2, lanes, segment_length, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L1"
    )
    L2 = mn.Link(
        1, lanes, segment_length, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L2"
    )
    O1 = mn.MeteredOnRamp(origin_capacities[0], name="O1")
    O2 = (
        mn.MeteredOnRamp(origin_capacities[1], name="O2")
        if control_O2_rate
        else mn.SimplifiedMeteredOnRamp(origin_capacities[1], "unlimited", name="O2")
    )
    D1 = mn.CongestedDestination(name="D1")
    net = (
        mn.Network()
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
    warns: bool = False,
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
    warn : bool, optional
        A flag to indicate whether a warning message should be raised when the error is
        found to be increasing, or when the maximum number of iterations are reached.

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
    err_previous = float("+inf")
    for k in range(maxiter):
        x0_ss = f(x0)
        err = float(np.linalg.norm(x0_ss - x0))
        if err < tol:
            return x0_ss, err, k
        elif err > err_previous:
            if warns:
                warn(
                    "Increasing error encountered in steady-state search "
                    f"({err} > {err_previous})."
                )
            return x0_ss, err, k

        x0 = x0_ss
        err_previous = err

    if warns:
        warn(
            f"Maximum number of iterations reached in steady-state search ({maxiter})."
        )
    return x0_ss, err, k
