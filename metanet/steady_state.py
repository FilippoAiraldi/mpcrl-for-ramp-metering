from typing import Callable, TypeVar
from warnings import warn

import numpy as np

Tx = TypeVar("Tx")
Tu = TypeVar("Tu")
Td = TypeVar("Td")


def get_steady_state(
    F: Callable[[Tx, Tu, Td], Tx],
    x0: Tx,
    u: Tu,
    d: Td,
    tol: float = 1e-3,
    maxiter: int = 500,
) -> tuple[Tx, float, int]:
    r"""Searches the steady-state of the given dynamics.

    Parameters
    ----------
    F : Callable[[Tx, Tu, Td], Tx]
        The dynamics function of the form `x+ = F(x, u, d)`, where `x` is the state, `u`
        the action, `d` the disturbance.
    x0 : Tx
        The initial steady-state guess. The type must be compatible with numpy.
    u : Tu
        The action at which to find the steady-state.
    d : Td
        The disturbance at which to find the steady-state.
    tol : float, optional
        Error tolerance for convergence, by default 1e-3
    maxiter : int, optional
        Maximum iterations, by default 500

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
        x0_ss = F(x0, u, d)
        err = float(np.linalg.norm(x0_ss - x0))  # type: ignore[operator]
        if err < tol:
            return x0_ss, err, k
        elif err >= err_previous:
            warn(
                "Increasing error encountered in steady-state search "
                f"({err} >= {err_previous})."
            )
            return x0_ss, err, k

        x0 = x0_ss
        err_previous = err

    warn(f"Maximum number of iterations reached in steady-state search ({maxiter}).")
    return x0_ss, err, k
