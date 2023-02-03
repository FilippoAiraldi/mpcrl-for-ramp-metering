from typing import Literal, Type, TypeVar

import casadi as cs
from mpcrl import Agent

from metanet.highway_traffic_env import HighwayTrafficEnv
from rl.common import _update_fixed_parameters, _wrap_agent
from util.constants import EnvConstants as EC
from util.constants import RlConstants as RC

SymType = TypeVar("SymType", cs.SX, cs.MX)
AgentType = TypeVar("AgentType", bound="HighwayTrafficPkAgent")


class HighwayTrafficPkAgent(Agent[SymType]):
    """Perfect-knowledge (PK) agent for the traffic control task, meaning that the agent
    has access to all the exact information underlying the environment."""

    __slots__ = ("_forecast_length",)

    def __init__(self, **kwargs) -> None:
        """Initializes the PK agent."""
        fixed_pars = {n: p for n, (p, _) in RC.parameters.items()}
        fixed_pars.update({"rho_crit": EC.rho_crit, "a": EC.a, "v_free": EC.v_free})
        kwargs["fixed_parameters"] = fixed_pars
        super().__init__(**kwargs)
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
