from logging import DEBUG, INFO
from typing import Literal

import numpy.typing as npt
from mpcrl import Agent
from mpcrl.wrappers.agents import Log, Wrapper

from metanet.highway_traffic_env import HighwayTrafficEnv
from util.constants import EnvConstants as EC


def _update_fixed_parameters(
    parameters: dict[str, npt.ArrayLike], env: HighwayTrafficEnv, forecast_length: int
) -> None:
    """Updates the internal demand as the forecasted demands, and the last action
    taken in the env. If the episode is over, deletes them instead."""
    if env.demand.exhausted:
        del parameters["d"], parameters["a-"]
    else:
        parameters["d"] = env.demand.forecast(forecast_length).T
        parameters["a-"] = env.last_action


def _wrap_agent(
    agent: Agent,
    verbose: Literal[0, 1, 2, 3],
) -> Wrapper:
    """Allows to build an instance of the agent that can be wrapped in the following
    wrappers (from inner to outer, where the outer returns last):
        - `Log`
    """
    if verbose > 0:
        level = INFO
        frequencies: dict[str, int] = {}
        excluded: list[str] = []
        if verbose >= 2:
            frequencies["on_episode_end"] = 1
            level = DEBUG
        if verbose >= 3:
            frequencies["on_env_step"] = int(EC.Tfin / EC.T / EC.steps)
        agent = Log(
            agent,
            level=level,
            log_frequencies=frequencies,
            exclude_mandatory=excluded,
        )
    return agent
