import argparse
from itertools import repeat
from math import exp

from util.constants import STEPS_PER_SCENARIO
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

    group = parser.add_argument_group("RL algorithm parameters")
    group.add_argument(
        "--agent-type",
        "--agent_type",
        type=str,
        choices=("pk", "lstdq"),
        help="Type of agent to simulate.",
        required=True,
    )
    group.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    group.add_argument(
        "--update-freq",
        "--update_freq",
        type=int,
        default=STEPS_PER_SCENARIO // 2,
        help="Update frequency of the learning agent (in terms of env-steps)",
    )
    group.add_argument(
        "--lr", type=float, default=1e-1, help="Learning rate of the agent."
    )
    group.add_argument(
        "--max-update",
        "--max_update",
        type=float,
        default=1 / 5,
        help="Maximum value parameters can be updated as percentage of current value.",
    )
    group.add_argument(
        "--replaymem-size",
        "--replaymem_size",
        type=int,
        default=STEPS_PER_SCENARIO * 10,
        help="Maximum size of the experience replay buffer.",
    )
    group.add_argument(
        "--replaymem-sample",
        "--replaymem_sample",
        type=float,
        default=0.5,
        help="Size of the replay memory samples as percentage of maximum replay size.",
    )
    group.add_argument(
        "--replaymem-sample-latest",
        "--replaymem_sample_latest",
        type=float,
        default=0.2,
        help="Size of the replay memory sample to dedicate to latest transitions.",
    )
    group.add_argument(
        "--exp-chance",
        "--exp_chance",
        type=float,
        default=0.1,
        help="Chance of exploration (epsilon-greedy strategy).",
    )
    group.add_argument(
        "--exp-strength",
        "--exp_strength",
        type=float,
        default=5e-2,  # 100 / 2000
        help="Strength of exploration.",
    )
    group.add_argument(
        "--exp-decay",
        "--exp_decay",
        type=float,
        default=exp(-1 / 5),
        help="Multiplicative decay rate of exploration chance and strength.",
    )

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
        default=2,
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

    args = parser.parse_args()

    # perform some checks and processing
    assert (
        0.0 <= args.replaymem_sample <= 1.0
    ), f"Replay sample size must be in [0,1]; got {args.replaymem_sample} instead."
    assert (
        0.0 <= args.exp_chance <= 1.0
    ), f"Chance of exploration must be in [0,1]; got {args.replaymem_sample} instead."
    assert (
        0.0 <= args.exp_decay <= 1.0
    ), f"Exploration decay must be in [0,1]; got {args.replaymem_sample} instead."
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
    if args.reduce <= 0:
        raise argparse.ArgumentTypeError("--reduce must be a positive integer.")
    return args
