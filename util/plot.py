from itertools import chain, cycle, repeat
from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from util import io
from util.constants import STEPS_PER_SCENARIO as K
from util.constants import EnvConstants as EC

MARKERS = ("o", "s", "v", "^", ">", "<", "P", "x", "D")
LINESTYLES = ("-", "--", "-.", ":")
OPTS = {"fill_between.alpha": 0.25}
NP_RANDOM = np.random.default_rng(0)
PARAM_LATEX = {
    "rho_crit": r"$\tilde{\rho}_{crit}$",
    "a": r"$\tilde{a}$",
    "v_free": r"$\tilde{v}_{free}$",
    "weight_tts": r"$\theta_T$",
    "weight_var": r"$\theta_V$",
    "weight_slack": r"$\theta_C$",
    "weight_init_rho": r"$\theta^\rho_\lambda$",
    "weight_init_v": r"$\theta^v_\lambda$",
    "weight_init_w": r"$\theta^w_\lambda$",
    "weight_stage_rho": r"$\theta^\rho_\ell$",
    "weight_stage_v": r"$\theta^v_\ell$",
    "weight_stage_w": r"$\theta^w_\ell$",
    "weight_terminal_rho": r"$\theta^\rho_\Gamma$",
    "weight_terminal_v": r"$\theta^v_\Gamma$",
    "weight_terminal_w": r"$\theta^w_\Gamma$",
}


def set_mpl_defaults() -> None:
    """Sets the default options for Matplotlib."""
    np.set_printoptions(precision=4)
    plt.style.use("bmh")  # 'seaborn-darkgrid'
    mpl.rcParams["lines.solid_capstyle"] = "round"
    mpl.rcParams["lines.linewidth"] = 1.5
    mpl.rcParams["lines.markersize"] = 2
    mpl.rcParams["savefig.dpi"] = 600


def _set_axis_opts(
    axs: Iterable[Axes],
    left=None,
    right=None,
    bottom=0,
    top=None,
    intx: bool = False,
    nbins: int = 6,
) -> None:
    """Internal utility to customize the x- and y-axis."""
    for ax in axs:
        ax.set_xlim(left=left, right=right)
        ax.set_ylim(bottom=bottom, top=top)
        if intx and not getattr(ax.xaxis.get_major_locator(), "_integer", False):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=nbins))


def _adjust_limits(axs: Iterable[Axes]) -> None:
    """Adjusts the axes limits (ensures all plotted data is shown)."""
    for ax in axs:
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.autoscale(enable=True, axis="y", tight=False)


def _add_title(fig: Figure, labels: Sequence[str]) -> None:
    """Adds a title to the figure."""
    if len(labels) == 1:
        return
    lss = cycle(LINESTYLES)
    fig.suptitle(" vs ".join(f"{lb} ({ls})" for lb, ls in zip(labels, lss)))


def _plot_population(
    ax: Axes,
    x: npt.NDArray,
    y: npt.NDArray,
    use_median: bool = False,
    log: bool = False,
    marker: Optional[str] = None,
    ls: Optional[str] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> None:
    """Internal utility to plot a quantity from some population of envs/agents."""
    y_avg = (np.nanmedian if use_median else np.nanmean)(y, 0)  # type: ignore[operator]
    y_std = 2 * np.nanstd(y, 0)
    method = ax.semilogy if log else ax.plot
    c = method(x, y_avg, label=label, marker=marker, ls=ls, color=color)[0].get_color()
    a = OPTS["fill_between.alpha"]
    ax.fill_between(x, y_avg - y_std, y_avg + y_std, alpha=a, color=c, label=None)


def _moving_average(x: np.ndarray, w: int, mode: str = "full") -> np.ndarray:
    """Computes the moving average of x (along axis=1) with window size w."""
    return np.asarray(
        [np.convolve(x[i], np.ones(w), mode) / w for i in range(x.shape[0])]
    )


def _save2tikz(*figs: Figure) -> None:
    """Saves the figure to a tikz file. See https://pypi.org/project/tikzplotlib/."""
    import tikzplotlib

    for fig in figs:
        mpl.lines.Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
        mpl.lines.Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
        mpl.legend.Legend._ncol = property(lambda self: self._ncols)
        tikzplotlib.save(
            f"figure_{fig.number}.tex",
            figure=fig,
            extra_axis_parameters={r"tick scale binop=\times"},
        )


def plot_traffic_quantities(
    envsdata: list[dict[str, npt.NDArray]], labels: list[str], paper: bool
) -> None:
    if paper:
        fig1, axs1 = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)

        for envsdatum in envsdata:
            # plot demands
            demands = envsdatum["demands"].reshape(-1, K, 3)
            time = np.arange(1, demands.shape[1] + 1) * EC.T * EC.steps * 60
            for i, (ax, lbl) in enumerate(
                [(axs1[0], "$O_1$"), (axs1[0], "$O_2$"), (axs1[1], "$D_1$")]
            ):
                _plot_population(ax, time, demands[..., i], ls="--", color=f"C{i}")
                idx = NP_RANDOM.integers(demands.shape[0])
                ax.plot(time, demands[idx, :, i], color=f"C{i}", label=lbl)

            # plot 1st, middle, and last on-ramp queues
            idx = [0, 1, 2, 6]
            O2_queue = envsdatum["state"][..., -1]
            time = np.arange(1, O2_queue.shape[2] + 1) * EC.T * EC.steps * 60
            ax2.axhline(y=EC.ramp_max_queue["O2"], color="k", ls="--", label=None)
            for i, j in enumerate(idx, start=3):
                lbl = f"Ep. {j + 1}"
                _plot_population(ax2, time, O2_queue[:, j], label=lbl, color=f"C{i}")

        _adjust_limits(chain(axs1, (ax2,)))
        axs1[0].set_ylabel("Entering flow (veh/h)")
        axs1[1].set_ylabel("Downstream density\n(veh/km/lane)")
        ax.set_xlabel("Time (min)")
        for ax in axs1:
            ax.legend(loc="upper right")
        ax2.set_ylabel("$O2$ queue (veh)")
        ax2.set_xlabel("Time (min)")
        ax2.legend(loc="upper right", ncol=1)

        _save2tikz(fig1, fig2)
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
            axs[i].set_xlabel("Time (h)")
        _adjust_limits(axs)
        _add_title(fig, labels)


def plot_costs(
    envsdata: list[dict[str, npt.NDArray]], labels: list[str], paper: bool
) -> None:
    fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)

    # process costs
    costnames = ("tts", "var", "cvi")
    ylbls = ("TTS", "Control variability", "Constraint violation")
    envscosts: list[np.ndarray] = []
    for envsdatum in envsdata:
        costs = np.stack([envsdatum[n].sum(2) for n in costnames], axis=-1)
        envscosts.append(costs)

    # make plotting
    for costs, ls in zip(envscosts, cycle(LINESTYLES)):
        ep = np.arange(1, costs.shape[1] + 1)
        for ylbl, cost, ax in zip(ylbls, np.rollaxis(costs, 2), axs):
            _plot_population(ax, ep, cost, ls=ls)
            ax.set_ylabel(ylbl)
            # ax.set_yscale("log")

    # set axis options
    axs[-1].set_xlabel("Learning episode")
    _adjust_limits(axs)
    _add_title(fig, labels)
    if paper:
        _save2tikz(fig)


def plot_agent_quantities(
    agentsdata: list[dict[str, npt.NDArray]], labels: list[str], paper: bool
) -> None:
    # sourcery skip: low-code-quality
    if paper:
        fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
        fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
        fig3, axs3 = plt.subplots(5, 3, constrained_layout=True, sharex=True)
        for agentsdatum in agentsdata:
            n_agents, n_episodes = agentsdatum["weight_tts"].shape[:2]
            td_errors = agentsdatum["td_errors"]

            # plot moving average of TD error
            timesteps_per_ep = td_errors.shape[1] // n_episodes
            td_ma = _moving_average(td_errors, timesteps_per_ep, "valid")
            td_ma = td_ma[:, :: EC.steps * 2]  # reduce number of elements to plot
            episodes = np.linspace(1, n_episodes, td_ma.shape[1])
            _plot_population(ax1, episodes, td_ma, log=True)

            # plot example of instantaneous TD error
            td_errors_per_ep = td_errors.reshape(n_agents, -1, timesteps_per_ep)
            time = np.arange(1, td_errors_per_ep.shape[2] + 1) * EC.T * EC.steps * 60
            ax2.plot(time, td_errors_per_ep[0, 3], "o")

            # TODO: plot some of the parameters more nicely

            # plot all the parameters for the appendix
            rows = [
                ("a", "rho_crit", "v_free"),
                ("weight_tts", "weight_var", "weight_slack"),
                ("weight_init_rho", "weight_init_v", "weight_init_w"),
                ("weight_stage_rho", "weight_stage_v", "weight_stage_w"),
                ("weight_terminal_rho", "weight_terminal_v", "weight_terminal_w"),
            ]
            # n_colors = len(plt.rcParams['axes.prop_cycle'])
            episodes = np.arange(1, n_episodes + 1)
            idx = np.arange(0, n_episodes, 2).tolist() + [n_episodes - 1]
            for row, axs in zip(rows, axs3):
                for par_name, ax in zip(row, axs):
                    ax.set_ylabel(PARAM_LATEX[par_name])
                    if par_name not in agentsdatum:
                        ax.set_axis_off()
                        continue
                    parameter = agentsdatum[par_name].reshape(n_agents, n_episodes, -1)
                    for i, p in enumerate(np.moveaxis(parameter, 2, 0)):
                        # ls = LINESTYLES[i // n_colors]
                        _plot_population(ax, episodes[idx], p[:, idx], color=f"C{i}")

        ax1.set_xlabel("Learning episode")
        ax1.set_ylabel(r"$\tau$")
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel(r"$\tau$")
        for ax in axs3[-1]:
            ax.set_xlabel("Learning episode")
        _adjust_limits(chain([ax1, ax2], axs3.flatten()))

        _save2tikz(fig1, fig2, fig3)
    else:

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
        a_key = td_errors_key = None
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
        weight_keys = sorted(weight_keys)
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
        plot_ = lambda v, ax: ax.hlines(
            v, *ax.get_xlim(), colors="darkgrey", zorder=-1e3
        )
        for val, ax in zip((EC.a, EC.rho_crit, EC.v_free), (a_ax, rho_ax, v_ax)):
            if ax is not None:
                plot_(val, ax)


def other_plots():
    # plot of Veq
    fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
    fns = [
        r"sims/sim_15_dynamics_a.xz",
        r"sims/sim_15_dynamics_a_rho_wo_track_higher_var.xz",
    ]
    lbls = [r"($a$)", r"($a, \rho_{crit}$)"]
    rho = np.linspace(0, 160, 300).reshape(-1, 1, 1)
    rho_ = rho.flatten()
    v_free_true = EC.v_free * np.exp(-1 / EC.a * np.power(rho / EC.rho_crit, EC.a))
    ax1.plot(rho_, v_free_true.flatten(), "k--", label=r"True $V_{eq}$")
    for (_, _, agentsdatum), lbl in zip(io.load_data(fns), lbls):
        n_agents, n_episodes = agentsdatum["a"].shape[:2]
        a = agentsdatum["a"].reshape(n_agents, n_episodes)
        if "rho_crit" in agentsdatum:
            rho_crit = agentsdatum["rho_crit"].reshape(n_agents, n_episodes)
        else:
            rho_crit = np.full((n_agents, n_episodes), 0.7 * EC.rho_crit)
        v_free = 1.3 * EC.v_free * np.exp(-1 / a * np.power(rho / rho_crit, a))
        _plot_population(ax1, rho_, v_free[..., 0].T, label=r"$V_{eq}$ Ep. 1 " + lbl)
        _plot_population(ax1, rho_, v_free[..., -1].T, label=r"$V_{eq}$ Ep. 80 " + lbl)
    _adjust_limits([ax1])
    ax1.legend()

    # plot comparison of performances of different parametrisations
    fig2, axs2 = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    fns = [
        r"sims/sim_15_no_dynamics.xz",
        r"sims/sim_15_dynamics_a.xz",
        r"sims/sim_15_dynamics_a_rho_wo_track.xz",
        r"sims/sim_15_dynamics_a_rho_with_track.xz",
        r"sims/sim_15_dynamics_a_v_wo_track.xz",
        r"sims/sim_15_dynamics_a_v_with_track.xz",
        r"sims/sim_15_dynamics_a_rho_v_wo_track.xz",
        r"sims/sim_15_dynamics_a_rho_v_with_track.xz",
    ]
    labels = [
        r"Not Learning Dynamics",
        r"Learning $a$",
        r"Learning $a$ and $\rho_{crit}$ (no tracking)",
        r"Learning $a$ and $\rho_{crit}$ (with tracking)",
        r"Learning $a$ and $v_{free}$ (no tracking)",
        r"Learning $a$ and $v_{free}$ (with tracking)",
        r"Learning $a$, $\rho_{crit}$ and $v_{free}$ (no tracking)",
        r"Learning $a$, $\rho_{crit}$ and $v_{free}$ (with tracking)",
    ]
    labels = iter(labels)
    costnames = ("tts", "var", "cvi")
    ylbls = ("TTS", "Control variability", "Constraint violation")
    envscosts: list[np.ndarray] = []
    for _, envsdatum, _ in io.load_data(fns):
        costs = np.stack([envsdatum[n].sum(2) for n in costnames], axis=-1)
        envscosts.append(costs)
    for costs, lbl in zip(envscosts, labels):
        ep = np.arange(1, costs.shape[1] + 1)
        for ylbl, cost, ax in zip(ylbls, np.rollaxis(costs, 2), axs2):
            ax.set_ylabel(ylbl)
            ax.plot(ep, np.nanmean(cost, 0), label=lbl if ylbl == "TTS" else None)
            if ylbl.startswith("Constraint"):
                ax.set_yscale("log")
    axs2[-1].set_xlabel("Learning episode")
    _adjust_limits(axs2)
    fig2.legend(loc="outside upper center", ncols=2)

    _save2tikz(fig1, fig2)
