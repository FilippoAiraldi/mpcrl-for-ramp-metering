from typing import Literal

from gymnasium import Env
from mpcrl import ExperienceReplay, LearningRate, RlLearningAgent, UpdateStrategy
from mpcrl import exploration as E
from mpcrl import schedulers as S

from metanet import HighwayTrafficEnv
from mpc import HighwayTrafficMpc
from rl import HighwayTrafficLstdQLearningAgent, HighwayTrafficPkAgent
from rl.agents import get_agent_components
from util import STEPS_PER_SCENARIO as K
from util import EnvConstants as EC


def evaluate_pk_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    discount_factor: float,
    demands_type: Literal["constant", "random"],
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
    demands_type : {"constant", "random"}
        Type of demands affecting the network.
    sym_type : {"SX", "MX"}
        The type of casadi symbols to use during the simulation.
    seed : int
        RNG seed for the simulation.
    verbose : {0, 1, 2, 3}
        Level of verbosity of the agent's logger.

    Returns
    -------
    Env
        The wrapped instance of the traffic environment
    """
    # create env for evaluation
    env = HighwayTrafficEnv.wrapped(
        demands_type=demands_type,
        sym_type=sym_type,
        n_scenarios=scenarios,
        normalize_rewards=False,
    )

    # create controller
    mpc = HighwayTrafficMpc(env, discount_factor, parametric_cost_terms=False)

    # initialize the agent with full knowledge of the env
    fixed_pars = get_agent_components(mpc.parameters, float("nan"))[1].value_as_dict
    fixed_pars.update({"rho_crit": EC.rho_crit, "a": EC.a, "v_free": EC.v_free})
    agent = HighwayTrafficPkAgent.wrapped(
        mpc=mpc,
        fixed_parameters=fixed_pars,
        name=f"PkAgent{agent_n}",
        verbose=verbose,
    )

    # launch evaluation
    agent.evaluate(env, episodes, seed=seed, raises=False)
    return env


def train_lstdq_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    update_freq: int,
    discount_factor: float,
    learning_rate: float,
    learning_rate_decay: float,
    exploration_chance: float,
    exploration_strength: float,
    exploration_decay: float,
    experience_replay_size: int,
    experience_replay_sample: float,
    experience_replay_sample_latest: float,
    max_percentage_update: float,
    demands_type: Literal["constant", "random"],
    sym_type: Literal["SX", "MX"],
    seed: int,
    verbose: Literal[0, 1, 2, 3],
) -> tuple[Env, RlLearningAgent]:
    """Launches a simulation for the training of a second-order LSTD Q-learning agent in
    the traffic control environment.

    Parameters
    ----------
    agent_n : int
        Number of the agent. Used to formulate its name.
    episodes : int
        Number of episodes to evaluate the agent for.
    scenarios : int
        Number of demands' scenarios per episode.
    update_freq : int
        Frequency (in terms of env steps) with which to update the agent's parameters.
    discount_factor : float
        Discount factor (used only in the MPC; no learning occurs in the PK agent).
    learning_rate : float
        The learning rate of the RL algorithm.
    learning_rate_decay : float
        The rate at which the learning rate decays (exponentially, after each update).
    exploration_chance : float
        Probability of exploration (epsilon-greedy strategy).
    exploration_strength : float
        Strength of the exploration perturbation.
    exploration_decay : float
        Rate at which exploration chance and strength decay (after each update).
    experience_replay_size : int
        Maximum size of the experience replay memory.
    experience_replay_sample : float
        Size of experience samples (in terms of percentage of max size) per update.
    experience_replay_sample_latest: float
        Size of experience sample to dedicate to latest transitions.
    max_percentage_update : float
        Maximum percentage parameters update at each RL update.
    demands_type : {"constant", "random"}
        Type of demands affecting the network.
    sym_type : {"SX", "MX"}
        The type of casadi symbols to use during the simulation.
    seed : int
        RNG seed for the simulation.
    verbose : {0, 1, 2, 3}
        Level of verbosity of the agent's logger.

    Returns
    -------
    Env and Agent
        The wrapped instances of both the traffic environment and the agent.
    """
    # create env for training
    env = HighwayTrafficEnv.wrapped(
        demands_type=demands_type,
        sym_type=sym_type,
        n_scenarios=scenarios,
        normalize_rewards=False,
    )

    # create controller
    mpc = HighwayTrafficMpc(env, discount_factor, parametric_cost_terms=True, seed=seed)

    # initialize the agent's components
    update_strategy = UpdateStrategy(update_freq, skip_first=K // update_freq)
    exploration = E.EpsilonGreedyExploration(
        epsilon=S.ExponentialScheduler(exploration_chance, exploration_decay),
        strength=S.ExponentialScheduler(exploration_strength, exploration_decay),
        hook="on_episode_end",
        seed=seed,
    )
    experience = ExperienceReplay(
        maxlen=experience_replay_size,
        sample_size=experience_replay_sample,
        include_latest=experience_replay_sample_latest,
        seed=seed,
    )
    fixed_pars, learnable_pars, lr = get_agent_components(mpc.parameters, learning_rate)
    agent = HighwayTrafficLstdQLearningAgent.wrapped(
        mpc=mpc,
        update_strategy=update_strategy,
        discount_factor=discount_factor,
        learning_rate=LearningRate(
            S.ExponentialScheduler(lr, learning_rate_decay), hook="on_episode_end"
        ),
        learnable_parameters=learnable_pars,
        fixed_parameters=fixed_pars,
        exploration=exploration,
        experience=experience,
        max_percentage_update=max_percentage_update,
        record_td_errors=True,
        name=f"LstdQAgent{agent_n}",
        verbose=verbose,
    )

    # launch training
    agent.train(env, episodes, seed=seed, raises=False)
    return env, agent
