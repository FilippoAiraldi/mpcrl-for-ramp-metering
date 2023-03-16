from logging import DEBUG, INFO
from typing import Literal, Type, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt
from mpcrl import Agent, LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.wrappers.agents import Log, RecordUpdates, Wrapper

from metanet import HighwayTrafficEnv
from util import EnvConstants as EC
from util import MpcRlConstants as MRC

SymType = TypeVar("SymType", cs.SX, cs.MX)
AgentType = TypeVar("AgentType", bound="HighwayTrafficPkAgent")


def _update_fixed_parameters(
    parameters: dict[str, npt.ArrayLike], env: HighwayTrafficEnv
) -> None:
    """Updates the internal demand forecasts and the last action taken in the env."""
    parameters["d"] = env.demand.forecast(MRC.prediction_horizon).T
    parameters["a-"] = env.last_action


def _wrap_agent(
    agent: Agent,
    record_updates: bool,
    verbose: Literal[0, 1, 2, 3],
) -> Wrapper:
    """Allows to build an instance of the agent that can be wrapped in the following
    wrappers (from inner to outer, where the outer returns last):
        - `RecordUpdates`
        - `Log`
    """
    if record_updates:
        agent = RecordUpdates(agent)
    if verbose > 0:
        level = INFO
        frequencies: dict[str, int] = {}
        if verbose >= 2:
            frequencies["on_episode_end"] = 1
            level = DEBUG
        if verbose >= 3:
            frequencies["on_env_step"] = int(EC.Tfin / EC.T / EC.steps)
        agent = Log(agent, level=level, log_frequencies=frequencies)
    return agent


def get_agent_components(
    mpc_parameters: dict[str, SymType], lr0: float
) -> tuple[
    dict[str, npt.ArrayLike], LearnableParametersDict[SymType], npt.NDArray[np.floating]
]:
    """Given the agent's (symbolic) parameters, gets
    - the fixed parameters
    - the learnable parameters
    - the learning rate vector"""
    fixed_pars, learnable_pars, lrs = {}, [], []
    for name, par in MRC.parameters.items():
        if par.learnable:
            if name not in mpc_parameters:
                continue
            symbol = mpc_parameters[name]
            size = symbol.size1()
            learnable_pars.append(
                LearnableParameter(name, size, par.value, *par.bounds, symbol)
            )
            lrs.append(np.full(size, lr0 * par.lr_multiplier))
        else:
            fixed_pars[name] = par.value
    return fixed_pars, LearnableParametersDict(learnable_pars), np.concatenate(lrs)


class HighwayTrafficPkAgent(Agent[SymType]):
    """Perfect-knowledge (PK) agent for the traffic control task, meaning that the agent
    has access to all the exact information underlying the environment."""

    def on_episode_start(self, env: HighwayTrafficEnv, episode: int) -> None:
        _update_fixed_parameters(self.fixed_parameters, env)
        super().on_episode_start(env, episode)

    def on_env_step(self, env: HighwayTrafficEnv, episode: int, timestep: int) -> None:
        _update_fixed_parameters(self.fixed_parameters, env)
        super().on_env_step(env, episode, timestep)

    @classmethod
    def wrapped(
        cls: Type[AgentType],
        verbose: Literal[0, 1, 2, 3],
        *agent_args,
        **agent_kwargs,
    ) -> AgentType:
        """Allows to build an instance of the agent that can be wrapped in the following
        wrappers (from inner to outer, where the outer returns last):
         - `Log`

        Parameters
        ----------
        cls : Type[AgentType]
            The type of env to instantiate.
        verbose : {0, 1, 2,  3}
            The level of verbosity for the logging wrapper.

        Returns
        -------
        AgentType
            Wrapped instance of the agent.
        """
        return _wrap_agent(cls(*agent_args, **agent_kwargs), False, verbose)


class HighwayTrafficLstdQLearningAgent(LstdQLearningAgent[SymType, float]):
    def on_episode_start(self, env: HighwayTrafficEnv, episode: int) -> None:
        _update_fixed_parameters(self.fixed_parameters, env)
        super().on_episode_start(env, episode)

    def on_env_step(self, env: HighwayTrafficEnv, episode: int, timestep: int) -> None:
        _update_fixed_parameters(self.fixed_parameters, env)
        super().on_env_step(env, episode, timestep)

    @classmethod
    def wrapped(
        cls: Type[AgentType],
        verbose: Literal[0, 1, 2, 3],
        *agent_args,
        **agent_kwargs,
    ) -> AgentType:
        """Allows to build an instance of the agent that can be wrapped in the following
        wrappers (from inner to outer, where the outer returns last):
         - `RecordUpdates`
         - `Log`

        Parameters
        ----------
        cls : Type[AgentType]
            The type of env to instantiate.
        verbose : {0, 1, 2,  3}
            The level of verbosity for the logging wrapper.

        Returns
        -------
        AgentType
            Wrapped instance of the agent.
        """
        return _wrap_agent(cls(*agent_args, **agent_kwargs), True, verbose)
