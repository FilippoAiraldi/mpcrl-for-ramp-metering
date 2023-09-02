import argparse

import matplotlib.pyplot as plt

from util import io, plot


def launch_visualization(args: argparse.Namespace):
    plot.set_mpl_defaults()
    names, envsdata, agentsdata = zip(*io.load_data(args.filenames))
    if args.traffic:
        plot.plot_traffic_quantities(envsdata, names, args.paper)
    if args.cost:
        plot.plot_costs(envsdata, names, args.paper)
    if args.agent:
        plot.plot_agent_quantities(agentsdata, names, args.paper)
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
        help="Plots all the available plots (-t, -c, -a).",
    )
    group.add_argument(
        "-P",
        "--paper",
        action="store_true",
        help="Plots figures for paper.",
    )

    args = parser.parse_args()
    if args.all:
        args.traffic = args.cost = args.agent = True
    del args.all

    launch_visualization(args)
