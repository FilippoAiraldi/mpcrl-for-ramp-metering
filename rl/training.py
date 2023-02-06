from typing import Literal

from gymnasium import Env

from metanet import HighwayTrafficEnv
from mpc import HighwayTrafficMpc
from rl import HighwayTrafficPkAgent
from util import EnvConstants as EC
from util import RlConstants as RC


def evaluate_pk_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    discount_factor: float,
    sym_type: Literal["SX", "MX"],
    seed: int,
    verbose: Literal[0, 1, 2, 3],
) -> Env:
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
        The wrapped instance of the traffic environment
    """
    # create env for evaluation
    env = HighwayTrafficEnv.wrapped(
        sym_type=sym_type,
        n_scenarios=scenarios,
        normalize_rewards=False,
    )

    # create controller
    mpc = HighwayTrafficMpc(env, discount_factor, parametric_cost_terms=False)

    # initialize the agent with full knowledge of the env
    fixed_parameters = get_learnable_parameters(mpc).value_as_dict
    fixed_parameters.update({"rho_crit": EC.rho_crit, "a": EC.a, "v_free": EC.v_free})
    agent = HighwayTrafficPkAgent.wrapped(
        mpc=mpc,
        fixed_parameters=fixed_parameters,
        name=f"PkAgent<{agent_n}>",
        verbose=verbose,
    )

    # launch evaluation
    agent.evaluate(
        env,
        episodes,
        seed=seed,
        raises=False,
    )
    return env


# def train_lstdq_agent(
#     agent_n: int,
#     episodes: int,
#     sym_type: Literal["SX", "MX"],
#     seed: int,
# ) -> Any:
#     raise NotImplementedError
