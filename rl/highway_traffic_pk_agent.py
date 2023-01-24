from typing import Optional, TypeVar

import casadi as cs
from mpcrl import Agent

from metanet.highway_traffic_env import HighwayTrafficEnv
from mpc.highway_traffic_mpc import HighwayTrafficMpc
from util.constants import EnvConstants as EC
from util.constants import RlConstants as RC

SymType = TypeVar("SymType", cs.SX, cs.MX)


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
        self._set_varying_pars(env)
        super().on_episode_start(env, episode)

    def on_env_step(self, env: HighwayTrafficEnv, episode: int, timestep: int) -> None:
        self._set_varying_pars(env)
        super().on_env_step(env, episode, timestep)

    def on_episode_end(
        self, env: HighwayTrafficEnv, episode: int, rewards: float
    ) -> None:
        del self.fixed_parameters["d"], self.fixed_parameters["a-"]
        super().on_episode_end(env, episode, rewards)

    def _set_varying_pars(self, env: HighwayTrafficEnv) -> None:
        """Sets the internal demand as the forecasted demands, and the last action taken
        in the env."""
        self.fixed_parameters["d"] = env.demand.forecast(self._forecast_length).T
        self.fixed_parameters["a-"] = env.last_action
