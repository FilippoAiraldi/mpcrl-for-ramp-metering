from logging import DEBUG, INFO
from typing import Dict, List, Literal, Optional, Type, TypeVar

import casadi as cs
from mpcrl import Agent
from mpcrl.wrappers.agents import Log

from metanet.highway_traffic_env import HighwayTrafficEnv
from mpc.highway_traffic_mpc import HighwayTrafficMpc
from util.constants import EnvConstants as EC
from util.constants import RlConstants as RC

SymType = TypeVar("SymType", cs.SX, cs.MX)
AgentType = TypeVar("AgentType", bound="HighwayTrafficPkAgent")


class HighwayTrafficPkAgent(Agent[SymType]):
    """Perfect-knowledge (PK) agent for the traffic control task, meaning that the agent
    has access to all the exact information underlying the environment."""

    __slots__ = ("_forecast_length",)

    def __init__(
        self, mpc: HighwayTrafficMpc[SymType], name: Optional[str] = None
    ) -> None:
        """Initializes the PK agent.

        Parameters
        ----------
        mpc : HighwayTrafficMpc[SymType]
            The MPC which the agent can use as policy provider.
        name : str, optional
            Name of the agent.
        """
        self._forecast_length = mpc.nlp.parameters["d"].shape[1]
        fixed_pars = {n: p for n, (p, _) in RC.parameters.items()}
        fixed_pars.update({"rho_crit": EC.rho_crit, "a": EC.a, "v_free": EC.v_free})
        super().__init__(mpc, fixed_pars, name=name)

    def on_episode_start(self, env: HighwayTrafficEnv, episode: int) -> None:
        self._update_fixed_pars(env)
        super().on_episode_start(env, episode)

    def on_env_step(self, env: HighwayTrafficEnv, episode: int, timestep: int) -> None:
        self._update_fixed_pars(env)
        super().on_env_step(env, episode, timestep)

    def _update_fixed_pars(self, env: HighwayTrafficEnv) -> None:
        """Updates the internal demand as the forecasted demands, and the last action
        taken in the env. If the episode is over, deletes them instead."""
        # TODO: check the timestep. If it is the final, delete the parameters
        if env.demand.exhausted:
            del self.fixed_parameters["d"], self.fixed_parameters["a-"]
        else:
            self.fixed_parameters["d"] = env.demand.forecast(self._forecast_length).T
            self.fixed_parameters["a-"] = env.last_action

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
        agent = cls(*agent_args, **agent_kwargs)
        if verbose > 0:
            level = INFO
            frequencies: Dict[str, int] = {}
            excluded: List[str] = []
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
