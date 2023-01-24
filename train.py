from datetime import datetime
from time import perf_counter
from typing import Any, Literal

import joblib as jl
from csnlp.util.io import save

from metanet import HighwayTrafficEnv
from mpc import HighwayTrafficMpc
from rl import HighwayTrafficPkAgent
from util import parse_args, tqdm_joblib


def eval_pk_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    discount_factor: float,
    sym_type: Literal["SX", "MX"],
    seed: int,
) -> Any:
    env = HighwayTrafficEnv.wrapped(
        sym_type=sym_type,
        n_scenarios=scenarios,
        normalize_rewards=False,
    )
    mpc = HighwayTrafficMpc(env, discount_factor, False)
    agent = HighwayTrafficPkAgent(mpc, name=f"PkAgent[{agent_n}]")
    # TODO: wrap agent
    agent.evaluate(
        env,
        episodes,
        seed=seed,
    )
    # TODO: what to return?
    raise NotImplementedError


# def train_lstdq_agent(
#     agent_n: int,
#     episodes: int,
#     sym_type: Literal["SX", "MX"],
#     seed: int,
# ) -> Any:
#     raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = perf_counter()
    if args.pk:
        func = lambda n: eval_pk_agent(
            agent_n=n,
            episodes=args.episodes,
            scenarios=args.scenarios,
            discount_factor=args.gamma,
            sym_type=args.sym_type,
            seed=args.seed + (args.episodes + 1) * n,
        )
    elif args.lstdq:
        raise NotImplementedError
        # func = lambda n: train_lstdq_agent(
        #     agent_n=n,
        #     episodes=args.episodes,
        #     sym_type=args.sym_type,
        #     seed=args.seed + (args.episodes + 1) * n,
        # )

    # launch simulations
    print(f"[Simulation {args.runname.upper()} started at {date}]\nArgs: {args}")
    with tqdm_joblib(desc="Simulation", total=args.agents):
        data = jl.Parallel(n_jobs=args.n_jobs)(
            jl.delayed(func)(i) for i in range(args.agents)
        )

    # save results
    print(f"[Simulated {args.agents} agents]")
    save(
        filename=args.runname,
        date=date,
        args=args,
        simtime=perf_counter() - start,
        data=data,
    )
