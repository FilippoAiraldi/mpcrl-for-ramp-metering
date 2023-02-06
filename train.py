from datetime import datetime
from time import perf_counter

from joblib import Parallel, delayed

from rl import evaluate_pk_agent, train_lstdq_agent
from util import parse_train_args, save_data, tqdm_joblib

if __name__ == "__main__":
    args = parse_train_args()
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
        fun = lambda n: train_lstdq_agent(
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
    print(f"[Simulated {args.agents} agents]")
    save_data(
        filename=args.runname,
        agent_type=args.agent_type,
        data=data,
        date=date,
        args=args.__dict__,
        simtime=perf_counter() - start,
        compression="lzma",
    )
