from datetime import datetime
from time import perf_counter
from typing import Any, Literal

from joblib import Parallel, delayed

from metanet import HighwayTrafficEnv
from mpc import HighwayTrafficMpc
from rl import HighwayTrafficPkAgent
from util import parse_train_args, tqdm_joblib, save_data


def eval_pk_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    discount_factor: float,
    sym_type: Literal["SX", "MX"],
    seed: int,
    verbose: Literal[0, 1, 2, 3],
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
    verbose : {0, 1, 2, 3}
        Level of verbosity of the agent's logger.

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
    agent = HighwayTrafficPkAgent.wrapped(
        mpc=mpc, name=f"PkAgent<{agent_n}>", verbose=verbose
    )
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
    if args.agent_type == "pk":
        fun = lambda n: eval_pk_agent(
            agent_n=n,
            episodes=args.episodes,
            scenarios=args.scenarios,
            discount_factor=args.gamma,
            sym_type=args.sym_type,
            seed=args.seed + (args.episodes + 1) * n,
            verbose=args.verbose,
        )
    elif args.agent_type == "lstdq":
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
    save_data(
        filename=args.runname,
        agent_type=args.agent_type,
        data=data,
        date=date,
        args=args.__dict__,
        simtime=perf_counter() - start,
        compression="matlab",
    )
