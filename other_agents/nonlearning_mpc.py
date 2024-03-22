from typing import Literal, Self

from gymnasium import Env
from mpcrl import Agent

from metanet import HighwayTrafficEnv
from mpc import HighwayTrafficMpc
from rl.agents import (
    SymType,
    _update_fixed_parameters,
    _wrap_agent,
    get_agent_components,
)
from util import MpcRlConstants as MRC


class HighwayTrafficAgent(Agent[SymType]):
    """A non-learning MPC agent for the traffic control task."""

    def on_episode_start(self, env: HighwayTrafficEnv, episode: int) -> None:
        _update_fixed_parameters(self.fixed_parameters, env)
        super().on_episode_start(env, episode)

    def on_env_step(self, env: HighwayTrafficEnv, episode: int, timestep: int) -> None:
        _update_fixed_parameters(self.fixed_parameters, env)
        super().on_env_step(env, episode, timestep)

    @classmethod
    def wrapped(
        cls: type[Self], verbose: Literal[0, 1, 2, 3], *agent_args, **agent_kwargs
    ) -> Self:
        """Allows to build an instance of the agent that can be wrapped in the following
        wrappers (from inner to outer, where the outer returns last):
         - `Log`

        Parameters
        ----------
        cls : Type[AgentType]
            The type of agent to instantiate.
        verbose : {0, 1, 2, 3}
            The level of verbosity for the logging wrapper.

        Returns
        -------
        AgentType
            Wrapped instance of the agent.
        """
        return _wrap_agent(cls(*agent_args, **agent_kwargs), False, verbose)


def eval_nonlearning_mpc_agent(
    agent_n: int,
    episodes: int,
    scenarios: int,
    discount_factor: float,
    demands_type: Literal["constant", "random"],
    sym_type: Literal["SX", "MX"],
    seed: int,
    verbose: Literal[0, 1, 2, 3],
) -> Env:
    """Launches a simulation that evaluates a non-learning MPC agent in the traffic
    control environment.

    Parameters
    ----------
    agent_n : int
        Number of the agent. Used to formulate its name.
    episodes : int
        Number of episodes to evaluate the agent for.
    scenarios : int
        Number of demands' scenarios per episode.
    discount_factor : float
        Discount factor.
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
        demands_type=demands_type, sym_type=sym_type, n_scenarios=scenarios
    )

    # create controller
    mpc = HighwayTrafficMpc(env, discount_factor, parametric_cost_terms=False)

    # initialize the agent with wrong knowledge of the env dynamics
    fixed_pars = get_agent_components(mpc.parameters, float("nan"))[1].value_as_dict
    fixed_pars.update((n, MRC.parameters[n].value) for n in ("rho_crit", "a", "v_free"))
    agent = HighwayTrafficAgent.wrapped(
        mpc=mpc,
        fixed_parameters=fixed_pars,
        name=f"Mpc-Agent{agent_n}",
        verbose=verbose,
    )

    # launch evaluation
    agent.evaluate(env, episodes, seed=seed, raises=False)
    return env
