from typing import Iterable, Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

OPTS = {
    "bar.edgecolor": "k",
    "bar.lw": 0.2,
    "bar.width": 0.8,
    "fill_between.alpha": 0.25,
    "plot.lw": 2.5,
}


def set_mpl_defaults() -> None:
    """Sets the default options for Matplotlib."""
    np.set_printoptions(precision=4)
    plt.style.use("bmh")  # 'seaborn-darkgrid'
    mpl.rcParams["lines.solid_capstyle"] = "round"
    mpl.rcParams["lines.linewidth"] = 3
    mpl.rcParams["savefig.dpi"] = 600


def _set_axis_opts(
    axs: Iterable[Axes],
    left=None,
    right=None,
    bottom=0,
    top=None,
    intx: bool = False,
    legend: bool = True,
    legendloc: str = "best",
) -> None:
    """Internal utility to customize the x- and y-axis."""
    for ax in axs:
        ax.set_xlim(left=left, right=right)
        ax.set_ylim(bottom=bottom, top=top)
        if intx and not getattr(ax.xaxis.get_major_locator(), "_integer", False):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if legend:
            ax.legend(loc=legendloc)


def _plot_population(
    ax: Axes,
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    use_median: bool = False,
    label: Optional[str] = None,
    method: Literal["fill_between", "errorbar"] = "fill_between",
) -> None:
    """Internal utility to plot a quantity from some population of envs/agents."""
    y_avg = (np.median if use_median else np.mean)(y, axis=0)  # type: ignore[operator]
    y_std = np.std(y, axis=0)
    if method == "fill_between":
        ax.fill_between(
            x,
            y_avg - y_std,
            y_avg + y_std,
            alpha=OPTS["fill_between.alpha"],
        )
        ax.plot(x, y_avg, label=label, lw=OPTS["plot.lw"])
    elif method == "errorbar":
        ax.errorbar(
            x,
            y_avg,
            y_std,
            label=label,
            errorevery=x.size // 10,
            capsize=5,
            lw=OPTS["plot.lw"],
        )
    else:
        raise ValueError(f"unsupported plotting method {method}.")


def plot_traffic_quantities(
    envsdata: dict[str, npt.NDArray[np.floating]],
    fig: Figure = None,
    label: Optional[str] = None,
    **kwargs,
) -> Figure:
    pass


def plot_costs(
    envsdata: dict[str, npt.NDArray[np.floating]],
    fig: Figure = None,
    label: Optional[str] = None,
    **_,
) -> Figure:
    if fig is None:
        fig, axs = plt.subplots(2, 2, constrained_layout=True, sharex=True)
        axs = axs.flatten()
    else:
        axs = fig.axes

    costnames = ("total", "tts", "var", "cvi")
    costs = np.stack([envsdata[n] for n in costnames[1:]], axis=-1)
    costs = costs.sum(2)  # sum costs per episodes
    J = costs.sum(-1, keepdims=True)  # tot cost per episode per agent
    all_costs = np.concatenate((J, costs), axis=-1)
    ep = np.arange(1, all_costs.shape[1] + 1)
    for costname, cost, ax in zip(costnames, np.rollaxis(all_costs, 2), axs):
        _plot_population(ax, ep, cost, label=label)
        ax.set_ylabel(f"{costname} cost")

    # set axis options
    _set_axis_opts(axs, intx=True)
    for i in (2, 3):
        axs[i].set_xlabel("Episode")
    return fig


