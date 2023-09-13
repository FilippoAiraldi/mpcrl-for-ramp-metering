from itertools import chain
from typing import Literal

import numpy as np
import numpy.typing as npt
from mpcrl.util.math import summarize_array
from numpy._typing import _ArrayLikeFloat_co
from scipy.signal import butter, filtfilt


class Demands:
    """Class containing the demands of the highway stretch at the mainstream origin O1,
    on-ramp O2, and the congestion at destination D1."""

    def __init__(self, demands: npt.NDArray[np.floating]):
        # demands : 3d array where
        #   - 1st dim: amount of iterations to simulate all the demands
        #   - 2nd dim: timesteps per iteration (the same step size used by MPC)
        #   - 3rd dim: number of demand quantities, i.e., 3
        assert demands.ndim == 3 and demands.shape[2] == 3, "Invalid demands array."
        self.demands = demands
        self.reset()

    def reset(self) -> None:
        """Resets the demands iteration."""
        self.t = 0

    @property
    def O1(self) -> npt.NDArray[np.floating]:
        """Gets the demands for the main origin O1."""
        return self.demands[:, :, 0]

    @property
    def O2(self) -> npt.NDArray[np.floating]:
        """Gets the demands for the on-ramp O2."""
        return self.demands[:, :, 1]

    @property
    def D1(self) -> npt.NDArray[np.floating]:
        """Gets the demands (congestion) for the destination D1."""
        return self.demands[:, :, 2]

    @property
    def exhausted(self) -> bool:
        """Gets whether the whole demands have been iterated through."""
        return self.t + 1 >= self.demands.shape[0]

    def forecast(self, length: int) -> npt.NDArray[np.floating]:
        """Gets the forecast of the demands from the current time up to the horizon of
        specified length. The forecast is flawless, and forecasted demands are padded to
        always have the expected size. This method does not increase the demands' time
        counter.

        Parameters
        ----------
        length : int
            Length of the forecast.

        Returns
        -------
        array of floats
            The future demands.
        """
        future_demands = self.demands[self.t : self.t + length]
        flattened = future_demands.reshape(-1, 3)
        if future_demands.shape[0] < length:
            gap = (length - future_demands.shape[0]) * self.demands.shape[1]
            flattened = np.append(flattened, flattened[-1, None].repeat(gap, 0), 0)
        return flattened

    def __getitem__(self, idx) -> npt.NDArray[np.floating]:
        return self.demands[idx]

    def __next__(self) -> npt.NDArray[np.floating]:
        d = self.demands[self.t]
        self.t += 1
        return d

    def __array__(self) -> npt.NDArray[np.floating]:
        return self.demands

    def __repr__(self) -> str:
        o1 = summarize_array(self.O1)
        o2 = summarize_array(self.O2)
        d1 = summarize_array(self.D1)
        return f"{self.__class__.__name__}(O1: {o1}, O2: {o2}, D1: {d1})"


def create_demand(
    time: _ArrayLikeFloat_co,
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    reps: int = 1,
) -> npt.NDArray[np.floating]:
    """Creates a demand profile over time that linearly interpolates between the given
    points `(x, y)` and is noisy (to  look more realistic).

    Parameters
    ----------
    time : 1D array_like
        The time vector.
    x : 1D array_like
        The x-coordinates where the profile is interpolated over.
    y : 1D array_like
        The y-coordinates where the profile is interpolated over.
    reps : int, optional
        Number of repetitions of the profile (see `numpy.tile`). Defaults to 1.

    Returns
    -------
    array of floats
        The interpolated demand profile.
    """
    demand = np.interp(time, x, y)
    if reps > 1:
        demand = np.tile(demand, reps)
    return demand


def create_demands(
    time: _ArrayLikeFloat_co,
    tf: float,
    reps: int = 1,
    steps_per_iteration: int = 1,
    kind: Literal["constant", "random"] = "constant",
    noise: tuple[float, float, float] = (100.0, 100.0, 2.5),
    seed: None | int | np.random.SeedSequence | np.random.Generator = None,
) -> Demands:
    """Creates the demands for the highway origins O1 and O2, and the destination D1.

    Parameters
    ----------
    time : array of floats
        The time vector.
    tf : float
        The final time of the scenario, i.e., how long a scenario is.
    reps : int, optional
        How many scenario repetitions to create, by default 1.
    steps_per_iteration : int, optional
        Reshapes the demands to return n steps per iteration, when iterated over.
    kind : "constant" or "random", optional
        If "constant", the scenarios are generated from constant data. If "random", the
        scenarios durations and levels are generated randomly. By default "constant".
    noise : 3-tuple of floats, optional
        The noise levels for the three demand scenarios, by default (100.0, 100.0, 2.5).
        This noise only partially affects the demand and is used to make them realistic.
    seed : None, int, seed sequence or generator, optional
        A random seed.

    Returns
    -------
    Demands
        The demands for origins O1 and O2, and destination D1.

    Raises
    ------
    ValueError
        Raises if `kind` is neither `"constant"` nor `"random"`.
    """
    np_random = np.random.default_rng(seed)
    L = np.asarray((1000, 500, 20), float)  # low demand levels
    H = np.asarray((3000, 1500, 60), float)  # high demand levels
    K = np.asarray(  # knee points in time
        [(0, 0.35, 1, 1.35), (0.15, 0.35, 0.6, 0.8), (0.5, 0.7, 1, 1.2)]
    )
    assert (K <= tf).all() and time[-1] <= tf, "invalid time specifications"

    if kind == "constant":
        D = np.stack(
            [
                create_demand(time, K[i], (L[i], H[i], H[i], L[i]), reps)
                for i in range(3)
            ],
            axis=-1,
        )

    elif kind == "random":
        time_ = np.concatenate([time + tf * r for r in range(reps)])
        alpha = 5 / 100
        L_ = np.tile(L, (reps, 2, 1))
        H_ = np.tile(H, (reps, 2, 1))
        K_ = np.concatenate([K + tf * r for r in range(reps)], -1)
        L_ += np_random.uniform(-L_ * alpha, L_ * alpha)
        H_ += np_random.uniform(-H_ * alpha, H_ * alpha)
        K_ += np_random.uniform(-tf * alpha / 2, +tf * alpha / 2, K_.shape)
        K_[:, 0] = np.maximum(K_[:, 0], 0)
        K_[:, -1] = np.minimum(K_[:, -1], tf * reps)
        assert (np.diff(K_) >= 0).all(), "internal error during demand generation"
        D = np.stack(
            [
                create_demand(
                    time_,
                    K_[i],
                    list(
                        chain.from_iterable(
                            (L_[r, 0, i], H_[r, 0, i], H_[r, 1, i], L_[r, 1, i])
                            for r in range(reps)
                        )
                    ),
                )
                for i in range(3)
            ],
            axis=-1,
        )
        b, a = butter(3, 0.1)
        D = filtfilt(
            b, a, D + np_random.normal(scale=noise, size=D.shape), axis=0, method="gust"
        )
        D = np.maximum(0, D)
    else:
        raise ValueError(f"Unrecognized demand kind '{kind}'.")

    # reshape and return
    D = D.reshape(-1, steps_per_iteration, 3)
    return Demands(D)
