from itertools import chain, repeat
from typing import Iterable, Literal, Optional

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from util.constants import EnvConstants as EC
from util.constants import STEPS_PER_SCENARIO as K

MARKERS = ("o", "s", "v")
LINESTYLES = ("-", "--", "-.")
OPTS = {
    "fill_between.alpha": 0.25,
    "plot.lw": 2,
}


def set_mpl_defaults() -> None:
    """Sets the default options for Matplotlib."""
    np.set_printoptions(precision=4)
    plt.style.use("bmh")  # 'seaborn-darkgrid'
    mpl.rcParams["lines.solid_capstyle"] = "round"
    mpl.rcParams["lines.linewidth"] = 2
    mpl.rcParams["lines.markersize"] = 3
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


def adjust_limits(figs: Iterable[Figure]) -> None:
    """Adjusts the axes limits in the figures (ensures all plotted data is shown)."""
    for fig in figs:
        axs: list[Axes] = fig.get_axes()
        for ax in axs:
            min_y, max_y = ax.dataLim.ymin, ax.dataLim.ymax
            bottom_y, top_y = ax.get_ylim()
            if min_y < bottom_y or max_y > top_y:
                ax.set_ylim(bottom=min_y, top=max_y)


def _plot_population(
    ax: Axes,
    x: npt.NDArray,
    y: npt.NDArray,
    use_median: bool = False,
    marker: Optional[str] = None,
    ls: Optional[str] = None,
    label: Optional[str] = None,
    method: Literal["fill_between", "errorbar"] = "fill_between",
    color: Optional[str] = None,
) -> None:
    """Internal utility to plot a quantity from some population of envs/agents."""
    y_avg = (np.nanmedian if use_median else np.nanmean)(y, 0)  # type: ignore[operator]
    y_std = np.nanstd(y, 0)
    if method == "fill_between":
        o = ax.plot(
            x, y_avg, label=label, marker=marker, ls=ls, lw=OPTS["plot.lw"], color=color
        )[0]
        ax.fill_between(
            x,
            y_avg - y_std,
            y_avg + y_std,
            alpha=OPTS["fill_between.alpha"],
            color=o.get_color(),
            label=None,
        )
    elif method == "errorbar":
        ax.errorbar(
            x,
            y_avg,
            y_std,
            label=label,
            marker=marker,
            ls=ls,
            lw=OPTS["plot.lw"],
            errorevery=x.size // 10,
            capsize=5,
            color=color,
        )
    else:
        raise ValueError(f"unsupported plotting method {method}.")


def plot_traffic_quantities(
    envsdata: dict[str, npt.NDArray[np.floating]],
    fig: Figure = None,
    label: Optional[str] = None,
    reduce: int = 1,
    **_,
) -> Figure:
    if fig is None:
        fig, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
        axs = axs.flatten()
    else:
        axs = fig.axes

    # reduce number of points to plot
    if reduce > 1:
        raxis = 2  # reduction axis
        envsdata_ = {
            k: v.take(range(0, v.shape[raxis], reduce), raxis)
            for k, v in envsdata.items()
        }
    else:
        envsdata_ = envsdata

    # flatten episodes along time
    envsdata_ = {
        k: v.reshape(v.shape[0], v.shape[1] * v.shape[2], -1)
        for k, v in envsdata_.items()
    }

    # make plots
    data = (
        *np.array_split(envsdata_["state"], 3, axis=-1),  # rho, v and w
        envsdata_["flow"][:, :, -2:],  # origin flow
        envsdata_["action"],
        envsdata_["demands"],
    )
    ylbls = (
        r"$\rho$ (veh/km/lane)",
        r"$v$ (km/h)",
        r"$w$ (veh)",
        r"$q_O$ (veh/h)",
        r"$a$ (veh/h)",
        r"$d$ (veh/h, veh/km/lane)",
    )
    for ax, datum, ylbl in zip(axs, data, ylbls):
        time = np.arange(datum.shape[1]) * EC.T * EC.steps * reduce
        datum = np.rollaxis(datum, 2)
        N = datum.shape[0]
        if N == 1:
            _plot_population(ax, time, datum[0], label=label)
        else:
            for datum_, ls in zip(datum, LINESTYLES):
                _plot_population(ax, time, datum_, ls=ls, label=label)
        ax.set_ylabel(ylbl)
    _set_axis_opts(axs)
    for i in (4, 5):
        axs[i].set_xlabel("time (h)")
    return fig
    # mean_demand = envsdata_["demands"].reshape(10, -1, 120, 3).reshape(-1, 120, 3)


def plot_costs(
    envsdata: dict[str, npt.NDArray[np.floating]],
    fig: Figure = None,
    label: Optional[str] = None,
    **_,
) -> Figure:
    if fig is None:
        fig = plt.figure(constrained_layout=True)
        G = gridspec.GridSpec(2, 2, figure=fig)
        axs = [fig.add_subplot(G[i]) for i in zip(*np.unravel_index(range(4), (2, 2)))]
    else:
        axs = fig.axes

    # group as: costs ∈ [n_agents, n_episodes, timesteps, type_of_costs]
    costnames = ("tts", "var", "cvi", "total")
    costs = np.stack([envsdata[n] for n in costnames[:-1]], axis=-1)

    # sum over time - either by episode or by scenario
    # costs = costs.sum(2)  # sum costs per episodes
    tax = 2  # time axis
    n_scenarios = np.ceil(costs.shape[tax] / K).astype(int)
    costs = np.concatenate(
        [sc.sum(tax, keepdims=True) for sc in np.array_split(costs, n_scenarios, tax)],
        axis=tax,
    ).reshape(-1, n_scenarios, 3)

    J = costs.sum(-1, keepdims=True)  # total cost per episode per agent
    all_costs = np.concatenate((costs, J), axis=-1)
    ep = np.arange(1, all_costs.shape[1] + 1)
    for costname, cost, ax in zip(costnames, np.rollaxis(all_costs, 2), axs):
        _plot_population(ax, ep, cost, label=label)
        ax.set_ylabel(costname)

    # set axis options
    _set_axis_opts(axs, intx=True)
    for ax in axs[-2:]:
        ax.set_xlabel("episode")
    return fig


def plot_agent_quantities(
    agentsdata: dict[str, npt.NDArray[np.floating]],
    fig: Figure = None,
    label: Optional[str] = None,
    reduce: int = 1,
    **_,
) -> Figure:
    # sourcery skip: low-code-quality

    def plot_parameters(key: str, ax: Axes, label: Optional[str] = None) -> None:
        n_agents, n_updates = agentsdata[key].shape[:2]
        updates = np.arange(n_updates)
        weight = np.rollaxis(agentsdata[key].reshape(n_agents, n_updates, -1), 2)
        N = weight.shape[0]
        if N == 1:
            _plot_population(ax, updates, weight[0], label=label)
        else:
            for w, ls in zip(weight, LINESTYLES):
                _plot_population(ax, updates, w, ls=ls, label=label)

    a_key, td_errors_key = None, None
    v_free_keys, rho_crit_keys, weight_keys = [], [], []
    for key in agentsdata:
        if key == "a":
            a_key = key
        elif key == "td_errors":
            td_errors_key = key
        elif key.startswith("v_free"):
            v_free_keys.append(key)
        elif key.startswith("rho_crit"):
            rho_crit_keys.append(key)
        elif key.startswith("weight_"):
            weight_keys.append(key)
        else:
            raise RuntimeError(f"unexpected key '{key}' in agent's data.")
    nplots = (
        (a_key is not None)
        + (td_errors_key is not None)
        + (len(v_free_keys) > 0)
        + (len(rho_crit_keys) > 0)
        + len(weight_keys)
    )
    assert nplots > 0, "No agent's quantities to plot."
    ncols = int(np.round(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))
    if fig is None:
        fig = plt.figure(constrained_layout=True)
        G = gridspec.GridSpec(nrows, ncols, figure=fig)
        axs = [
            fig.add_subplot(G[idx])
            for idx in zip(*np.unravel_index(range(nplots), (nrows, ncols)))
        ]
    else:
        axs = fig.axes
    axs_iter = iter(axs)

    # plot a
    if a_key is not None:
        ax = next(axs_iter)
        plot_parameters(a_key, ax, f"a ({label})")
        ax.set_ylabel(r"$a$")

    # plot rho_crit
    if rho_crit_keys:
        ax = next(axs_iter)
        for rho_crit_key in rho_crit_keys:
            plot_parameters(rho_crit_key, ax, label=f"{rho_crit_key} ({label})")
        ax.hlines(
            EC.rho_crit, *ax.get_xlim(), colors="k", ls="--", lw=OPTS["plot.lw"] / 2
        )
        ax.set_ylabel(r"$\rho_{crit}$")

    # plot v_free
    if v_free_keys:
        ax = next(axs_iter)
        for v_free_key in v_free_keys:
            plot_parameters(v_free_key, ax, label=f"{v_free_key} ({label})")
        ax.hlines(
            EC.v_free, *ax.get_xlim(), colors="k", ls="--", lw=OPTS["plot.lw"] / 2
        )
        ax.set_ylabel(r"$v_{free}$")

    # plot other weights
    for weight_key in weight_keys:
        ax = next(axs_iter)
        plot_parameters(weight_key, ax)
        ax.set_ylabel(weight_key.replace("_", " "), fontsize=9)

    # plot td_error
    if td_errors_key is not None:
        ax = next(axs_iter)
        td_errors = agentsdata[td_errors_key][:, ::reduce]
        time = np.arange(td_errors.shape[1]) * EC.T * EC.steps * reduce
        _plot_population(ax, time, td_errors, label=label)
        ax.set_ylabel(r"$\delta$")

    # set all axes' options
    _set_axis_opts(axs, bottom=None, intx=True, legend=False)

    # set x labels in bottom axes
    nxlbls = ncols - (ncols * nrows - nplots)
    xlbls = chain(
        (("update" if td_errors_key is None else "time (h)"),),
        repeat("update", nxlbls - 1),
    )
    for ax, xlbl in zip(reversed(axs), xlbls):
        ax.set_xlabel(xlbl)
    return fig
