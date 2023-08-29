import argparse
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from util import load_data, plot


def load_all_data(
    filenames: Iterable[str],
) -> Iterator[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]]:
    for i, fn in enumerate(filenames):
        name = Path(fn).stem
        data = load_data(fn)
        envsdata = data.pop("envs")
        agentsdata = data.pop("agents", {})
        details = (f" -{k}: {v}" for k, v in data.items())
        print(f"p{i}) {name}", *details, sep="\n")
        yield name, envsdata, agentsdata


def launch_visualization(args: argparse.Namespace):
    plot.set_mpl_defaults()
    names, envsdata, agentsdata = zip(*load_all_data(args.filenames))
    if args.traffic:
        plot.plot_traffic_quantities(envsdata, names, reduce=args.reduce)
    if args.cost:
        plot.plot_costs(envsdata, names)
    if args.agent:
        plot.plot_agent_quantities(agentsdata, names, reduce=args.reduce)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launches visualization for different MPC-based agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Data")
    group.add_argument(
        "filenames", type=str, nargs="+", help="Simulation data to be visualized."
    )
    group = parser.add_argument_group("Plots")
    group.add_argument(
        "-t",
        "--traffic",
        action="store_true",
        help="Plots the traffic-related quantities.",
    )
    group.add_argument(
        "-c",
        "--cost",
        action="store_true",
        help="Plots the stage cost the agent witnessed during the simulation.",
    )
    group.add_argument(
        "-a",
        "--agent",
        action="store_true",
        help="Plots agent-specific quantities.",
    )
    group.add_argument(
        "-A",
        "--all",
        action="store_true",
        help="Plots all the available plots.",
    )
    group = parser.add_argument_group("Others")
    group.add_argument(
        "-r",
        "--reduce",
        type=int,
        default=1,
        help="Step-size to reduce the amount of points to plot.",
    )

    args = parser.parse_args()
    if args.all:
        args.traffic = args.cost = args.agent = True
    del args.all
    if args.reduce <= 0:
        raise argparse.ArgumentTypeError("--reduce must be a positive integer.")

    launch_visualization(args)
