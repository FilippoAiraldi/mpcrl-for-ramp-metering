import logging
from typing import Literal

import numpy as np
from gymnasium import Env

from metanet import HighwayTrafficEnv
from mpc.highway_traffic_mpc import _find_index_of_ramp_and_segment
from util import EnvConstants as EC
from util import MpcRlConstants as MRC


def _get_logger(name: str, verbose: Literal[0, 1, 2, 3]) -> logging.Logger:
    """Internal utility to quickly build a logger."""
    logger = logging.getLogger(name)
    if verbose <= 0:
        logger.disabled = True
        return logger
    level = logging.INFO if verbose <= 2 else logging.DEBUG
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(name)s@%(asctime)s> %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def eval_pi_alinea_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    gains: tuple[float, float],
    queue_management: bool,
    demands_type: Literal["constant", "random"],
    sym_type: Literal["SX", "MX"],
    seed: int,
    verbose: Literal[0, 1, 2, 3],
) -> Env:
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
    gains : tuple[float, float]
        PI-ALINEA gains (proportional, and integral).
    queue_management : bool
        Whether to apply queue management strategy to avoid exceeding maximum queue.
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
        The wrapped instances of the traffic environment.
    """
    # create env for training
    env = HighwayTrafficEnv.wrapped(
        demands_type=demands_type,
        sym_type=sym_type,
        n_scenarios=scenarios,
        normalize_rewards=False,
    )

    # define some agent parameters
    name = f"Pi-Alinea{agent_n}"
    Kp, Ki = gains
    downstread_density_desired = MRC.parameters["rho_crit"].value
    ramp = "O2"
    i_ramp, i_seg = _find_index_of_ramp_and_segment(env.network, ramp)
    w_max = EC.ramp_max_queue[ramp]

    # evaluate for each episode
    logger = _get_logger(name, verbose)
    seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))
    for episode, current_seed in zip(range(episodes), seeds):
        # reset for new episode
        truncated = terminated = False
        timestep = 0
        rewards = 0.0
        state, _ = env.reset(seed=current_seed)
        downstream_density_prev = state[i_seg]
        action = env.last_action.item()  # steady-state ramp flow, i.e., action

        # start new episode  loop
        while not (truncated or terminated):
            # compute PI-ALINEA control
            downstream_density = state[i_seg]
            action += Ki * (downstread_density_desired - downstream_density) - Kp * (
                downstream_density - downstream_density_prev
            )

            # add queue management strategy to avoid exceeding maximum queue
            if queue_management and timestep > 0:
                queue = state[env.n_segments * 2 + i_ramp]
                inflow = env.demand[timestep - 1, 0, i_ramp]
                restricted_action = (queue - w_max) / EC.T + inflow
                action = max(action, restricted_action)

            # apply control to environment
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, r, truncated, terminated, _ = env.step(action)

            # log timestep end
            logger.debug(f"env stepped in episode {episode} at time {timestep}.")
            rewards += r
            timestep += 1
            downstream_density_prev = downstream_density

        # log episode end
        logger.info(f"episode {episode} ended with rewards={rewards:.3f}.")
    return env
