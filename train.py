from datetime import datetime
from time import perf_counter
from typing import Any, Literal

from csnlp.util.io import save
import joblib as jl

from metanet import HighwayTrafficEnv
from mpc import HighwayTrafficMpc
from util import parse_args, tqdm_joblib


def eval_pk_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    sym_type: Literal["SX", "MX"],
    seed: int,
) -> Any:
    # TODO: wrap env
    env = HighwayTrafficEnv(sym_type, scenarios)
    mpc = HighwayTrafficMpc(env)
    # TODO: agent = ... function of mpc ...
    # TODO: wrap agent
    # TODO: call agent.train ...
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
