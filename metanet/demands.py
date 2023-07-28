from typing import Literal, Union

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
    reps: int = 1,
    steps_per_iteration: int = 1,
    kind: Literal["constant", "random"] = "constant",
    noise: tuple[float, float, float] = (100.0, 100.0, 2.5),
    seed: Union[None, int, np.random.SeedSequence, np.random.Generator] = None,
) -> Demands:
    """Creates the demands for the highway origins O1 and O2, and the destination D1.

    Parameters
    ----------
    time : array of floats
        _description_
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

    if kind == "constant":
        data = (
            ((0.00, 0.35, 1.0, 1.35), (1000, 3000, 3000, 1000)),  # demand at O1
            ((0.15, 0.35, 0.6, 0.80), (500, 1500, 1500, 500)),  # demand at O2
            ((0.50, 0.70, 1.0, 1.20), (20, 60, 60, 20)),  # congestion at D1
        )
        o1 = create_demand(time, *data[0], reps)
        o2 = create_demand(time, *data[1], reps)
        d1 = create_demand(time, *data[2], reps)

    elif kind == "random":
        lows = np_random.uniform((900, 400, 20), (1200, 600, 30), (reps + 1, 3))
        highs = np_random.uniform((2750, 1250, 45), (3500, 2000, 80), (reps, 3))
        centers = np_random.uniform((0.5, 0.3, 0.7), (0.9, 0.7, 1.0), (reps, 3))
        # NOTE: max width_low must fit in time[-1]
        widths_high = np_random.uniform((0.1, 0.05, 0.05), (0.2, 0.2, 0.25), (reps, 3))
        widths_low = np_random.uniform(widths_high * 1.5, widths_high * 3)

        o1_, o2_, d1_ = [], [], []
        for i in range(reps):
            x = np.stack(
                (
                    centers[i] - widths_low[i],
                    centers[i] - widths_high[i],
                    centers[i] + widths_high[i],
                    centers[i] + widths_low[i],
                ),
                axis=-1,
            )
            y = np.stack((lows[i], highs[i], highs[i], lows[i + 1]), axis=-1)
            o1_.append(create_demand(time, x[0], y[0]))
            o2_.append(create_demand(time, x[1], y[1]))
            d1_.append(create_demand(time, x[2], y[2]))
        o1 = np.concatenate(o1_)
        o2 = np.concatenate(o2_)
        d1 = np.concatenate(d1_)
    else:
        raise ValueError(f"Unrecognized demand kind '{kind}'.")

    # apply noise and positivity
    D = np.stack((o1, o2, d1), axis=-1)
    b, a = butter(3, 0.1)
    D = filtfilt(
        b, a, D + np_random.normal(scale=noise, size=D.shape), axis=0, method="gust"
    )
    D = np.maximum(0, D)

    # reshape and return
    D = D.reshape(-1, steps_per_iteration, 3)
    return Demands(D)
