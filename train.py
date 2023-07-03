from datetime import datetime
from time import perf_counter
import argparse
from math import exp

from joblib import Parallel, delayed

from rl import evaluate_pk_agent, train_lstdq_agent
from util import save_data, tqdm_joblib
from util.constants import STEPS_PER_SCENARIO
from util.runs import get_runname


def launch_training(args: argparse.Namespace) -> None:
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = perf_counter()
    if args.agent_type == "pk":
        fun = lambda n: evaluate_pk_agent(
            agent_n=n,
            episodes=args.episodes,
            scenarios=args.scenarios,
            discount_factor=args.gamma,
            sym_type=args.sym_type,
            seed=args.seed + (args.episodes + 1) * n,
            verbose=args.verbose,
        )
    elif args.agent_type == "lstdq":
        fun = lambda n: train_lstdq_agent(  # type: ignore[assignment,return-value]
            agent_n=n,
            episodes=args.episodes,
            scenarios=args.scenarios,
            update_freq=args.update_freq,
            discount_factor=args.gamma,
            learning_rate=args.lr,
            exploration_chance=args.exp_chance,
            exploration_strength=args.exp_strength,
            exploration_decay=args.exp_decay,
            experience_replay_size=args.replaymem_size,
            experience_replay_sample=args.replaymem_sample,
            experience_replay_sample_latest=args.replaymem_sample_latest,
            max_percentage_update=args.max_update,
            sym_type=args.sym_type,
            seed=args.seed + (args.episodes + 1) * n,
            verbose=args.verbose,
        )

    # launch simulations
    print(f"[Simulation {args.runname.upper()} started at {date}]\nArgs: {args}")
    with tqdm_joblib(desc="Simulation", total=args.agents):
        data = Parallel(n_jobs=args.n_jobs)(delayed(fun)(i) for i in range(args.agents))

    # save results
    print(f"[Simulated {args.runname.upper()} with {args.agents} agents]")
    save_data(
        filename=args.runname,
        agent_type=args.agent_type,
        data=data,
        date=date,
        args=args.__dict__,
        simtime=perf_counter() - start,
        compression="lzma",
    )


if __name__ == "__main__":
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
        default="SX",
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

    # launch simulation
    launch_training(args)
