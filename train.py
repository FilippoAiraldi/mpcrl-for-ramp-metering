from datetime import datetime
from time import perf_counter
from typing import Any, Literal

from joblib import delayed, Parallel
from csnlp.util.io import save

from metanet import HighwayTrafficEnv
from mpc import HighwayTrafficMpc
from rl import HighwayTrafficPkAgent
from util import parse_train_args, tqdm_joblib


def eval_pk_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    discount_factor: float,
    sym_type: Literal["SX", "MX"],
    seed: int,
) -> Any:
    """Launches a simulation that evaluates a perfect-knowledge (PK) agent in the
    traffic control environment.

    Parameters
    ----------
    agent_n : int
        Number of the agent. Used to formulate its name.
    episodes : int
        Number of episodes to evaluate the agent for.
    scenarios : int
        Number of demands' scenarios per episode.
    discount_factor : float
        Discount factor (used only in the MPC; no learning occurs in the PK agent).
    sym_type : {"SX", "MX"}
        The type of casadi symbols to use during the simulation.
    seed : int
        RNG seed for the simulation.

    Returns
    -------
    Any
        _description_
    """
    env = HighwayTrafficEnv.wrapped(
        sym_type=sym_type,
        n_scenarios=scenarios,
        normalize_rewards=False,
    )
    mpc = HighwayTrafficMpc(env, discount_factor, False)
    agent = HighwayTrafficPkAgent(mpc, name=f"PkAgent[{agent_n}]")
    agent.evaluate(
        env,
        episodes,
        seed=seed,
    )
    return env


# def train_lstdq_agent(
#     agent_n: int,
#     episodes: int,
#     sym_type: Literal["SX", "MX"],
#     seed: int,
# ) -> Any:
#     raise NotImplementedError


if __name__ == "__main__":
    args = parse_train_args()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = perf_counter()
    if args.pk:
        fun = lambda n: eval_pk_agent(
            agent_n=n,
            episodes=args.episodes,
            scenarios=args.scenarios,
            discount_factor=args.gamma,
            sym_type=args.sym_type,
            seed=args.seed + (args.episodes + 1) * n,
        )
    elif args.lstdq:
        raise NotImplementedError
        # fun = lambda n: train_lstdq_agent(
        #     agent_n=n,
        #     episodes=args.episodes,
        #     sym_type=args.sym_type,
        #     seed=args.seed + (args.episodes + 1) * n,
        # )

    # launch simulations
    print(f"[Simulation {args.runname.upper()} started at {date}]\nArgs: {args}")
    with tqdm_joblib(desc="Simulation", total=args.agents):
        data = Parallel(n_jobs=args.n_jobs)(delayed(fun)(i) for i in range(args.agents))

    # save results
    print(f"[Simulated {args.agents} agents]")
    save(
        filename=args.runname,
        date=date,
        args=args,
        simtime=perf_counter() - start,
        data=data,
    )
