from typing import Callable, TypeVar
from warnings import warn

import numpy as np

Tx = TypeVar("Tx")


def get_steady_state(
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
