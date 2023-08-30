from itertools import chain, cycle, repeat
from typing import Iterable, Literal, Optional, Sequence

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from util.constants import STEPS_PER_SCENARIO as K
from util.constants import EnvConstants as EC

MARKERS = ("o", "s", "v", "^", ">", "<", "P", "x", "D")
LINESTYLES = ("-", "--", "-.")
OPTS = {"fill_between.alpha": 0.25}


def set_mpl_defaults() -> None:
    """Sets the default options for Matplotlib."""
    np.set_printoptions(precision=4)
    plt.style.use("bmh")  # 'seaborn-darkgrid'
    mpl.rcParams["lines.solid_capstyle"] = "round"
    mpl.rcParams["lines.linewidth"] = 1.25
    mpl.rcParams["lines.markersize"] = 2
    mpl.rcParams["savefig.dpi"] = 600


def _set_axis_opts(
    axs: Iterable[Axes],
    left=None,
    right=None,
    bottom=0,
    top=None,
    intx: bool = False,
) -> None:
    """Internal utility to customize the x- and y-axis."""
    for ax in axs:
        ax.set_xlim(left=left, right=right)
        ax.set_ylim(bottom=bottom, top=top)
        if intx and not getattr(ax.xaxis.get_major_locator(), "_integer", False):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))


def _adjust_limits(axs: Iterable[Axes]) -> None:
    """Adjusts the axes limits (ensures all plotted data is shown)."""
    for ax in axs:
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=False)


def _add_title(fig: Figure, labels: Sequence[str]) -> None:
    """Adds a title to the figure."""
    if len(labels) == 1:
        return
    fig.suptitle(" vs ".join(f"{lb} ({ls})" for lb, ls in zip(labels, LINESTYLES)))


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
        o = ax.plot(x, y_avg, label=label, marker=marker, ls=ls, color=color)[0]
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
            errorevery=x.size // 10,
            capsize=5,
            color=color,
        )
    else:
        raise ValueError(f"unsupported plotting method {method}.")


def _save2tikz(*figs: Figure) -> None:
    """Saves the figure to a tikz file. See https://pypi.org/project/tikzplotlib/."""
    import tikzplotlib

    for fig in figs:
        for ax in fig.axes:
            for child in ax.get_children():
                if isinstance(child, mpl.legend.Legend):
                    child._ncol = child._ncols
        tikzplotlib.save(
            f"figure_{fig.number}.tex",
            figure=fig,
            extra_axis_parameters={r"tick scale binop=\times"},
        )


def plot_traffic_quantities(
    envsdata: list[dict[str, npt.NDArray[np.floating]]],
    labels: list[str],
    paper: bool = False,
) -> None:
    if paper:
        envsdatum = envsdata[0]

        # plot demands
        np_random = np.random.default_rng(0)
        fig1, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        scenarios_per_episode = int(envsdatum["demands"].shape[2] / K)
        all_demands = envsdatum["demands"].reshape(-1, K, 3)
        all_demands_avg = np.tile(all_demands.mean(0), (scenarios_per_episode, 1))
        all_demands_std = np.tile(all_demands.std(0), (scenarios_per_episode, 1))
        n_agents, n_episodes = envsdatum["demands"].shape[:2]
        time = np.arange(all_demands_avg.shape[0]) * EC.T * EC.steps * 60
        for i, (ax, lbl) in enumerate(
            [(axs[0], "$O_1$"), (axs[0], "$O_2$"), (axs[1], "$D_1$")]
        ):
            ax.fill_between(
                time,
                all_demands_avg[:, i] - 2 * all_demands_std[:, i],
                all_demands_avg[:, i] + 2 * all_demands_std[:, i],
                alpha=OPTS["fill_between.alpha"],
                color=f"C{i}",
                label=None,
            )
            j, k = np_random.integers((n_agents, n_episodes))
            ax.plot(time, envsdatum["demands"][j, k, :, i], color=f"C{i}", label=lbl)
        _adjust_limits(axs)
        axs[0].set_ylabel("Entering flow (veh/h)")
        axs[1].set_ylabel("Downstream density\n(veh/km/lane)")
        ax.set_xlabel("time (min)")
        [ax.legend(loc="upper right") for ax in axs]
        _save2tikz(fig1)
    else:
        fig, axs_ = plt.subplots(3, 2, constrained_layout=True, sharex=True)
        axs: Sequence[Axes] = axs_.flatten()

        # flatten episodes along time
        envsdata = [
            {
                k: v.reshape(v.shape[0], v.shape[1] * v.shape[2], -1)
                for k, v in envsdatum.items()
            }
            for envsdatum in envsdata
        ]

        # make plots
        ylbls = (
            r"$\rho$ (veh/km/lane)",
            r"$v$ (km/h)",
            r"$w$ (veh)",
            r"$q_O$ (veh/h)",
            r"$a$ (veh/h)",
            r"$d$ (veh/h, veh/km/lane)",
        )
        for envsdatum, ls in zip(envsdata, LINESTYLES):
            plotdata = (
                *np.array_split(envsdatum["state"], 3, axis=-1),  # rho, v and w
                envsdatum["flow"][:, :, -2:],  # origin flow
                envsdatum["action"],
                envsdatum["demands"],
            )
            for ax, datum, ylbl in zip(axs, plotdata, ylbls):
                time = np.arange(datum.shape[1]) * EC.T * EC.steps
                datum = np.rollaxis(datum, 2)
                N = datum.shape[0]
                if N == 1:
                    _plot_population(ax, time, datum[0], ls=ls)
                else:
                    for d_ in datum:
                        _plot_population(ax, time, d_, ls=ls)
                ax.set_ylabel(ylbl)

        # adjust some opts
        _set_axis_opts(axs)
        for i in (4, 5):
            axs[i].set_xlabel("time (h)")
        _adjust_limits(axs)
        _add_title(fig, labels)


def plot_costs(
    envsdata: list[dict[str, npt.NDArray[np.floating]]],
    labels: list[str],
    paper: bool = False,
) -> None:
    fig, axs = plt.subplots(1, 3, constrained_layout=True, sharex=True)

    # process costs
    costnames = ("tts", "var", "cvi")
    envscosts: list[np.ndarray] = []
    for envsdatum in envsdata:
        # group as: costs ∈ [n_agents, n_episodes, timesteps, cost_types]
        costs = np.stack([envsdatum[n] for n in costnames], axis=-1)

        # sum over time - either by episode or by scenario
        # costs = costs.sum(2)  # sum costs per episodes
        tax = 2  # time axis
        n_scenarios = np.ceil(costs.shape[tax] / K).astype(int)
        splitted_costs = np.array_split(costs, n_scenarios, tax)
        costs = np.concatenate(
            [c.sum(tax, keepdims=True) for c in splitted_costs], axis=tax
        ).reshape(costs.shape[0], -1, costs.shape[-1])

        # append to list
        envscosts.append(costs)

    # make plotting
    for costs, ls in zip(envscosts, LINESTYLES):
        ep = np.arange(costs.shape[1]) / n_scenarios
        for costname, cost, ax in zip(costnames, np.rollaxis(costs, 2), axs):
            _plot_population(ax, ep, cost, ls=ls)
            ax.set_ylabel(costname)

    # set axis options
    _set_axis_opts(axs, intx=True)
    for ax in axs:
        ax.set_xlabel("episode")
    _adjust_limits(axs)
    _add_title(fig, labels)


def plot_agent_quantities(
    agentsdata: list[dict[str, npt.NDArray[np.floating]]], labels: list[str]
) -> None:
    # sourcery skip: low-code-quality

    def plot_pars(
        datum: dict[str, npt.NDArray[np.floating]],
        keys: list[str],
        ax: Axes,
        ls: str,
        legend: bool = False,
    ) -> None:
        # prepare some quantities
        markers = cycle(MARKERS)
        labels_and_markers = []

        # loop over keys
        for key, marker in zip(keys, markers):
            if key not in datum:
                continue

            # rearrange data
            weight = datum[key]
            n_agents, n_updates, size1, size2 = weight.shape
            N = size1 * size2
            updates = np.arange(n_updates)
            w = np.rollaxis(weight.reshape(n_agents, n_updates, N), 2)

            # plot
            if N == 1:
                if len(keys) == 1:
                    marker = None  # type: ignore[assignment]
                else:
                    marker = next(markers)
                    labels_and_markers.append((key, marker))
                _plot_population(ax, updates, w[0], marker=marker, ls=ls)
            else:
                for i, (w_, marker) in enumerate(zip(w, markers)):
                    _plot_population(ax, updates, w_, marker=marker, ls=ls)
                    labels_and_markers.append((f"{key}_{i}", marker))

        # add legend
        if legend and labels_and_markers:
            ax.legend(
                handles=[
                    Line2D([], [], color="k", marker=marker, label=lbl)
                    for lbl, marker in labels_and_markers
                ]
            )

    # decide what to plot
    a_key, td_errors_key = None, None
    v_free_keys, rho_crit_keys, weight_keys = [], [], []
    for key in set(chain.from_iterable(agentsdata)):
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
    fig = plt.figure(constrained_layout=True)
    G = gridspec.GridSpec(nrows, ncols, figure=fig)
    axs: list[Axes] = [
        fig.add_subplot(G[idx])
        for idx in zip(*np.unravel_index(range(nplots), (nrows, ncols)))
    ]

    # make plots
    a_ax, rho_ax, v_ax = None, None, None
    for agentsdatum, ls in zip(agentsdata, LINESTYLES):
        axs_iter = iter(axs)

        # plot a
        if a_key is not None:
            a_ax = next(axs_iter)
            plot_pars(agentsdatum, [a_key], a_ax, ls, legend=True)
            a_ax.set_ylabel(r"$a$")

        # plot rho_crit
        if rho_crit_keys:
            rho_ax = next(axs_iter)
            plot_pars(agentsdatum, rho_crit_keys, rho_ax, ls, legend=True)
            rho_ax.set_ylabel(r"$\rho_{crit}$")

        # plot v_free
        if v_free_keys:
            v_ax = next(axs_iter)
            plot_pars(agentsdatum, v_free_keys, v_ax, ls, legend=True)
            v_ax.set_ylabel(r"$v_{free}$")

        # plot other weights
        for weight_key in weight_keys:
            ax = next(axs_iter)
            plot_pars(agentsdatum, [weight_key], ax, ls)
            ax.set_ylabel(weight_key.replace("_", " "), fontsize=9)

        # plot td_error
        if td_errors_key is not None:
            ax = next(axs_iter)
            # TODO: decide which TD error plot is better
            # td_errors = _moving_average(agentsdatum[td_errors_key], K)
            td_data = agentsdatum[td_errors_key]
            n_agents, n_timesteps = td_data.shape
            n_scenarios_per_episode = next(
                i for i in range(1, 11) if n_timesteps % (K * i - 1) == 0
            )
            td_errors = np.nanmean(
                td_data.reshape(n_agents, -1, n_scenarios_per_episode * K - 1), -1
            )
            episodes = np.arange(td_errors.shape[1])
            _plot_population(ax, episodes, td_errors, ls=ls)
            ax.set_ylabel(r"$\delta$")

    # set axis options
    _set_axis_opts(axs, bottom=None, intx=True)
    # set x labels in bottom axes
    nxlbls = ncols - (ncols * nrows - nplots)
    xlbls = chain(
        (("update" if td_errors_key is None else "episode"),),
        repeat("update", nxlbls - 1),
    )
    for ax, xlbl in zip(reversed(axs), xlbls):
        ax.set_xlabel(xlbl)
    _adjust_limits(axs)
    _add_title(fig, labels)

    # plot true parameters
    plot_ = lambda v, ax: ax.hlines(v, *ax.get_xlim(), colors="darkgrey", zorder=-1e3)
    for val, ax in zip((EC.a, EC.rho_crit, EC.v_free), (a_ax, rho_ax, v_ax)):
        if ax is not None:
            plot_(val, ax)
