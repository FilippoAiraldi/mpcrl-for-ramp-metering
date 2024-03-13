import argparse
from datetime import datetime
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed

from other_agents import eval_mpc_agent, eval_pi_alinea_agent
from rl import train_lstdq_agent
from util import save_data, tqdm_joblib
from util.constants import STEPS_PER_SCENARIO
from util.runs import get_runname


def launch_training(args: argparse.Namespace) -> None:
    seeds = np.random.SeedSequence(args.seed).generate_state(args.agents)
    if args.agent_type == "lstdq":

        def fun(n: int):
            return train_lstdq_agent(
                agent_n=n,
                episodes=args.episodes,
                scenarios=args.scenarios,
                update_freq=args.update_freq,
                discount_factor=args.gamma,
                learning_rate=args.lr,
                learning_rate_decay=args.lr_decay,
                exploration_chance=args.exp_chance,
                exploration_strength=args.exp_strength,
                exploration_decay=args.exp_decay,
                experience_replay_size=args.replaymem_size,
                experience_replay_sample=args.replaymem_sample,
                experience_replay_sample_latest=args.replaymem_sample_latest,
                max_percentage_update=args.max_update,
                demands_type=args.demands_type,
                sym_type=args.sym_type,
                seed=seeds[n],
                verbose=args.verbose,
            )

    elif args.agent_type == "mpc":

        def fun(n: int):
            return eval_mpc_agent(
                agent_n=n,
                episodes=args.episodes,
                scenarios=args.scenarios,
                discount_factor=args.gamma,
                demands_type=args.demands_type,
                sym_type=args.sym_type,
                seed=seeds[n],
                verbose=args.verbose,
            )

    elif args.agent_type == "pi-alinea":

        def fun(n: int):
            return eval_pi_alinea_agent(
                agent_n=n,
                episodes=args.episodes,
                scenarios=args.scenarios,
                gains=(args.Kp, args.Ki),
                queue_management=args.queue_management,
                demands_type=args.demands_type,
                sym_type=args.sym_type,
                seed=seeds[n],
                verbose=args.verbose,
            )[0]

    else:
        raise ValueError(f"unknown agent type {args.agent_type}")

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = perf_counter()
    print(f"[Simulation {args.runname.upper()} started at {date}]\nArgs: {args}")
    with tqdm_joblib(desc="Simulation", total=args.agents):
        data = Parallel(n_jobs=args.n_jobs)(delayed(fun)(i) for i in range(args.agents))

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
    parser = argparse.ArgumentParser(
        description="Launches simulation for different traffic agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Algorithm parameters")
    group.add_argument(
        "--agent-type",
        "--agent_type",
        type=str,
        choices=("lstdq", "mpc", "pi-alinea", "ddpg"),
        help="Type of agent to simulate.",
        required=True,
    )

    group = parser.add_argument_group("MPC-RL parameters")
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
        "--lr-decay", "--lr_decay", type=float, default=1.0, help="Learning rate decay."
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
        default=5e-2,
        help="Strength of exploration.",
    )
    group.add_argument(
        "--exp-decay",
        "--exp_decay",
        type=float,
        default=0.5,
        help="Multiplicative decay rate of exploration chance and strength.",
    )

    group = parser.add_argument_group("PI-ALINEA parameters")
    group.add_argument(
        "--Kp", type=float, default=32.07353865774536, help="Proportional gain."
    )
    group.add_argument(
        "--Ki", type=float, default=0.5419114131900662, help="Integral gain."
    )
    group.add_argument(
        "--queue-management",
        "--queue_management",
        action="store_true",
        help="Use a queue management stratey to avoid queue exceeding a max length.",
    )

    group = parser.add_argument_group("Simulation details")
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
    group.add_argument(
        "--demands-type",
        "--demands_type",
        type=str,
        choices=("constant", "random"),
        default="constant",
        help="Type of demands affecting the network.",
    )
    group = parser.add_argument_group("Others")
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

    args_ = parser.parse_args()
    assert (
        0.0 <= args_.replaymem_sample <= 1.0
    ), f"Replay sample size must be in [0,1]; got {args_.replaymem_sample} instead."
    assert (
        0.0 <= args_.exp_chance <= 1.0
    ), f"Chance of exploration must be in [0,1]; got {args_.replaymem_sample} instead."
    assert (
        0.0 <= args_.exp_decay <= 1.0
    ), f"Exploration decay must be in [0,1]; got {args_.replaymem_sample} instead."
    assert (
        args_.Kp >= 0 and args_.Ki >= 0
    ), f"PI-ALINEA gains must be non-negative; got {args_.Kp} and {args_.Ki} instead."
    args_.runname = get_runname(candidate=args_.runname)
    if args_.agents == 1:
        args_.n_jobs = 1  # don't parallelize

    launch_training(args_)
