from logging import DEBUG, INFO
from typing import Literal, Type, TypeVar

import casadi as cs
import numpy.typing as npt
from mpcrl import Agent, LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.wrappers.agents import Log, Wrapper

from metanet import HighwayTrafficEnv
from mpc import HighwayTrafficMpc
from util import EnvConstants as EC
from util import RlConstants as RC

SymType = TypeVar("SymType", cs.SX, cs.MX)
AgentType = TypeVar("AgentType", bound="HighwayTrafficPkAgent")


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


def get_fixed_parameters() -> dict[str, npt.ArrayLike]:
    """Gets the fixed (non-learnable) parameters."""
    return {
        n: v for n, (v, is_learnable, _) in RC.parameters.items() if not is_learnable
    }


def get_learnable_parameters(
    mpc: HighwayTrafficMpc[SymType],
) -> LearnableParametersDict[SymType]:
    """Gets the learnable parameters."""
    pars = mpc.parameters

    def get_par(name, value, bnds):
        sym = pars[name]
        return LearnableParameter(name, sym.size1(), value, *bnds, sym)

    return LearnableParametersDict(
        (
            get_par(name, value, bnds)
            for name, (value, is_learnable, bnds) in RC.parameters.items()
            if is_learnable and name in pars
        )
    )


class HighwayTrafficPkAgent(Agent[SymType]):
    """Perfect-knowledge (PK) agent for the traffic control task, meaning that the agent
    has access to all the exact information underlying the environment."""

    __slots__ = ("_forecast_length",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._forecast_length = self.V.nlp.parameters["d"].shape[1]

    def on_episode_start(self, env: HighwayTrafficEnv, episode: int) -> None:
        _update_fixed_parameters(self.fixed_parameters, env, self._forecast_length)
        super().on_episode_start(env, episode)

    def on_env_step(self, env: HighwayTrafficEnv, episode: int, timestep: int) -> None:
        _update_fixed_parameters(self.fixed_parameters, env, self._forecast_length)
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
        return _wrap_agent(cls(*agent_args, **agent_kwargs), verbose=verbose)
