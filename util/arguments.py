import argparse
from itertools import repeat

from util.runs import get_runname


def parse_train_args() -> argparse.Namespace:
    """Parses the arguments needed to run the highway traffic control environment and
    its (learning) agents.

    Returns
    -------
    argparse.Namespace
        The argument namespace.
    """

    # construct parser
    parser = argparse.ArgumentParser(
        description="Launches simulation for different MPC-based RL agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_argument_group("Agent type")
    group.add_argument(
        "--agent-type",
        "--agent_type",
        type=str,
        choices=("pk", "lstdq"),
        help="Type of agent to simulate.",
        required=True,
    )

    group = parser.add_argument_group("RL algorithm parameters")
    group.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    # group.add_argument(  # [3e-2, 3e-2, 1e-3, 1e-3, 1e-3],
    #     "--lr",
    #     type=float,
    #     nargs="+",
    #     default=[0.498],
    #     help="Learning rate. Can be a single float, or a list of floats. In "
    #     "the latter case, either one float per parameter name, or one "
    #     "per parameter element (in case of parameters that are vectors).",
    # )
    # group.add_argument(
    #     "--perturbation-decay",
    #     "--perturbation_decay",
    #     type=float,
    #     default=0.885,
    #     help="Rate at which the exploration (random perturbations of the MPC "
    #     "objective ) decays both in term of chance and strength.",
    # )
    # group.add_argument(
    #     "--max-perc-update",
    #     "--max_perc_update",
    #     type=float,
    #     default=float("inf"),
    #     help="Limits the maximum value that each parameter can be updated by "
    #     "a percentage of the current value.",
    # )

    group = parser.add_argument_group("Simulation length")
    group.add_argument(
        "--agents", type=int, default=1, help="Number of agent to simulate."
    )
    group.add_argument(
        "--episodes", type=int, default=10, help="Number of training episodes."
    )
    group.add_argument(
        "--scenarios",
        type=int,
        default=6,
        help="Number of demands' scenarios per training episode.",
    )

    group = parser.add_argument_group("Simulation details")
    group.add_argument(
        "--sym-type",
        "--sym_type",
        type=str,
        choices=("SX", "MX"),
        default="MX",
        help="Type of CasADi symbolic variable.",
    )
    group.add_argument(
        "--runname", type=str, default=None, help="Name of the simulation run."
    )
    group.add_argument("--seed", type=int, default=1909, help="RNG seed.")
    group.add_argument(
        "--n-jobs", "--n_jobs", type=int, default=-1, help="Simulation parallel jobs."
    )
    group.add_argument(
        "--verbose", type=int, choices=(0, 1, 2, 3), default=1, help="Verbosity level."
    )

    # group = parser.add_argument_group("RL experience replay parameters")
    # group.add_argument(
    #     "--replay-mem-size",
    #     "--replay_mem_size",
    #     type=int,
    #     default=1,
    #     help="How many epochs the replay memory can store.",
    # )
    # group.add_argument(
    #     "--replay-mem-sample",
    #     "--replay_mem_sample",
    #     type=float,
    #     default=1.0,
    #     help="Size of the replay memory samples (percentage).",
    # )

    args = parser.parse_args()

    # perform some checks and processing
    args.runname = get_runname(candidate=args.runname)
    if args.agents == 1:
        args.n_jobs = 1  # don't parallelize
    return args


def parse_visualization_args() -> argparse.Namespace:
    """Parses the arguments needed for the visualization of results obtained from the
    highway traffic control environment.

    Returns
    -------
    argparse.Namespace
        The argument namespace.
    """

    # construct parser
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

    # remove duplicate but keep order
    args.filenames = list(dict(zip(args.filenames, repeat(None))))

    if args.all:
        args.traffic = args.cost = args.agent = True
    del args.all
    if not (args.traffic or args.cost or args.agent):
        raise argparse.ArgumentError(
            None, "No plot selected for visualization; see help for -t, -c, -a and -A."
        )
    if args.reduce <= 0:
        raise argparse.ArgumentTypeError("--reduce must be a positive integer.")
    return args
