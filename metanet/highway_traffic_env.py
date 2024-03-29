from functools import cached_property
from typing import Any, Literal, SupportsFloat, TypeVar

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import sym_metanet
from csnlp.util.io import SupportsDeepcopyAndPickle
from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeReward
from mpcrl.wrappers.envs import MonitorInfos

from metanet.costs import get_stage_cost
from metanet.demands import create_demands
from metanet.network import get_network, steady_state
from util.constants import EnvConstants as EC

EnvType = TypeVar("EnvType", bound="HighwayTrafficEnv")


class HighwayTrafficEnv(
    gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    SupportsDeepcopyAndPickle,
):
    """
    ## Description

    This environment simulates a highway traffic network according to the METANET
    modelling framework.

    ## Action Space

    The action is an array containing only one element, which can take any non-negative
    value.

    ## Observation Space

    The observation (also state) space is an array of shape `(ns * 2 + no,)`, where `ns`
    is the number of segments and `no` the number of origins in the network. The state
    is then obtained via concatenation as
    ```
                            s = [rho^T, v^T, w^T]^T
    ```
    where `rho ∈ R^ns`, `v ∈ R^ns` and `w ∈ R^no`.

    ## Rewards/Costs

    The reward here is intended as a cost, and must be minized. It is computed according
    to the current state and action, and reflects
     - the time-spent on average by cars in traffic
     - the control input variability
     - the violation of maximum queues at the ramps.

    ## Demands and Starting State

    The demands (a.k.a., the disturbances) at the network's origin and destinations can
    random- or constant- generated at each reset, and the initial starting state depends
    on them.

    ## Episode End

    Each episode ends when all the demand scenarios (specified in the constructor) have
    been simulated.
    """

    def __init__(
        self,
        demands_type: Literal["constant", "random"],
        sym_type: Literal["SX", "MX"],
        n_scenarios: int,
        store_demands: bool = True,
    ) -> None:
        """Initializes the environment.

        Parameters
        ----------
        demands_type : "constant" or "random"
            Type of demand to generate at each reset.
        sym_type : "SX" or "MX"
            The type of CasADi symbolic variable to use.
        n_scenarios : int
            Number demand scenarios to generate after each reset (consequently, this
            affects also the length of each env's episode).
        store_demands : bool, optional
            Whether to store past demands in memory or not, by default `True`.
        """
        gym.Env.__init__(self)
        SupportsDeepcopyAndPickle.__init__(self)
        self.sym_type = sym_type
        self.demands_type = demands_type
        self.n_scenarios = n_scenarios
        self.time = np.arange(0.0, EC.Tscenario, EC.T)

        # create dynamics
        self.network, sympars = get_network(
            segment_length=EC.segment_length,
            lanes=EC.lanes,
            origin_capacities=EC.origin_capacities,
            rho_max=EC.rho_max,
            sym_type=sym_type,
        )
        self.network.step(
            init_conditions={self.network.origins_by_name["O1"]: {"r": 1}},
            T=EC.T,
            tau=EC.tau,
            eta=EC.eta,
            kappa=EC.kappa,
            delta=EC.delta,
            positive_next_speed=True,  # because they are prone to get negative
        )
        self.dynamics: cs.Function = sym_metanet.engine.to_function(
            net=self.network, T=EC.T, parameters=sympars, more_out=True, compact=2
        )
        self.dynamics_mapaccum = self.dynamics.mapaccum(EC.steps)
        self.realpars = {n: getattr(EC, n) for n in sympars}
        # NOTE: the dynamics are of the form
        #           Function(F:(x[8],u,d[3],p[3])->(x+[8],q[5])
        # where the inputs are
        #           x[8] = [rho[3], v[3], w[2]]     (3 segments, 2 origins)
        #           u = q_O2
        #           d = [d_O1, d_O2, d_D1]          (2 demands, 1 congestion)
        #           p = ...see `sympars`...
        # and the outpus are
        #           x+[8] = ...same as x...
        #           q[5] = [q[3], q_o[2]]           (3 segments, 2 origins)

        # set observation and action spaces
        ns = self.ns
        na = self.na
        self.observation_space = Box(0.0, np.inf, (ns,), np.float64)
        self.action_space = Box(0.0, np.inf, (na,), np.float64)

        # set reward/cost ranges and functions
        self.reward_range = (0.0, float("inf"))
        self.stage_cost = get_stage_cost(
            self.network,
            na,
            EC.T,
            {self.network.origins_by_name[n]: v for n, v in EC.ramp_max_queue.items()},
        )

        # create initial solution to steady-state search (used in reset)
        n_segments, n_origins = self.n_segments, self.n_origins
        rho0, v0, w0 = 10, 100, 0
        self._last_initial_state = np.asarray(
            [rho0] * n_segments + [v0] * n_segments + [w0] * n_origins
        )

        # initialize storage for past demands
        self.demands: list[npt.NDArray[np.floating]] | None = (
            [] if store_demands else None
        )

    @cached_property
    def ns(self) -> int:
        """Gets the number of states/observations in the environment."""
        return self.dynamics.size1_in(0)

    @cached_property
    def na(self) -> int:
        """Gets the number of actions in the environment."""
        return self.dynamics.size1_in(1)

    @cached_property
    def nd(self) -> int:
        """Gets the number of disturbances in the environment."""
        return self.dynamics.size1_in(2)

    @cached_property
    def n_segments(self) -> int:
        """Gets the number of segments in all links of the network."""
        return sum(link.N for _, _, link in self.network.links)

    @cached_property
    def n_origins(self) -> int:
        """Gets the number of origins across the network."""
        return len(self.network.origins)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.

        Parameters
        ----------
        seed : int, optional
            RNG seed.
        options : dict, optional
            A dict with optional indications for resetting

             - `demands_kind`: the type of demand to generate (`'constant'` or
             `'random'`). Defaults to `'random'`.

             - `demands_noise`: a 3-tuple indicating the noise levels for demands at O1,
             O2 and D1. Defaults to `(100, 100, 2.5)`.

             - `steady_state_x0`: the initial state for which to search the steady-state
             for the newly generated demand. Defaults to the previous result of the same
             function.

             - `last_action`: the last action, i.e., before the env reset used to
             compute the steady-state. Defaults to setting the ramp to fully opened.

             - `steady_state_tol`: tolerance for steady-state, `1e-3` by default.

             - `steady_state_maxiter`: maximum iterations for steady-state, `500` by
             default.


        Returns
        -------
        tuple of array and dict[str, Any]
            A tuple containing the initial state/observation and some info.
        """
        if options is None:
            options = {}

        # reset seeds
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        # create demands (and record them in storage)
        self.demand = create_demands(
            time=self.time,
            tf=EC.Tscenario,
            reps=self.n_scenarios,
            steps_per_iteration=EC.steps,
            kind=self.demands_type,
            noise=options.get("demands_noise", (95.0, 95.0, 1.7)),
            seed=self.np_random,
        )
        if self.demands is not None:
            # NOTE: do not save all demands (only 0) to reduce size of results
            self.demands.append(self.demand[:, 0])

        # compute initial state
        x0 = options.get("steady_state_x0", self._last_initial_state)
        d = cs.DM(self.demand[0, 0])
        u = options.get("last_action", d[1])
        p = cs.DM(self.realpars.values())
        f = lambda x: self.dynamics(x, u, d, p)[0].full().ravel()
        state, err, iters = steady_state(
            f=f,
            x0=x0,
            tol=options.get("steady_state_tol", 1e-3),
            maxiter=options.get("steady_state_maxiter", 500),
            warns=options.get("warns", False),
        )
        assert self.observation_space.contains(state), "Invalid reset state."
        self._last_initial_state = state  # save to warmstart next reset steady-state
        self.state = state.reshape(-1, 1).repeat(EC.steps, 1)

        # now get the actual control action sufficient for the steady-state state
        self.last_action = self.dynamics(state, u, d, p)[1][-1].full().reshape(self.na)
        return state, {"steady_state_error": err, "steady_state_iters": iters}

    def step(
        self,
        action: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], SupportsFloat, bool, bool, dict[str, Any]]:
        a = np.asarray(action).reshape(self.na)
        assert self.action_space.contains(a), "Invalid action passed to step."
        s = self.state

        # compute cost of current state L(s,a) (actually, over the last bunch of states)
        # NOTE: since the action is only applied every EC.steps, penalize var_ only once
        tts_, var_, cvi_ = self.stage_cost(s, a, self.last_action)
        tts = np.sum(tts_).item()
        var = EC.stage_cost_weights["var"] * float(var_[0])
        cvi = EC.stage_cost_weights["cvi"] * np.sum(cvi_).item()
        cost = EC.stage_cost_weights["tts"] * tts + var + cvi

        # step the dynamics
        d = cs.DM(next(self.demand).T)
        s_next, flows = self.dynamics_mapaccum(s[:, -1], a, d, self.realpars.values())

        # save next state and add information in dict to be saved
        # NOTE: save only last state and flow for sake of reducing size of results; and
        # don't save the scaled tts, since we want to see its real (unscaled) value
        self.state = np.maximum(0, s_next)
        self.state[(self.state < 0.0) & np.isclose(self.state, 0.0)] = 0.0
        self.last_action = a
        observation = self.state[:, -1]
        assert self.observation_space.contains(observation), "Invalid state after step."
        info: dict[str, Any] = {
            "state": s[:, -1],
            "action": a,
            "flow": flows.full()[:, -1],
            "tts": tts,
            "var": var,
            "cvi": cvi,
        }
        return observation, cost, False, self.demand.exhausted, info

    @classmethod
    def wrapped(
        cls: type[EnvType],
        monitor_deques_size: int | None = None,
        normalize_rewards: bool = True,
        normalization_gamma: float = 0.99,
        *env_args,
        **env_kwargs,
    ) -> EnvType:
        """Allows to build an instance of the env that can be wrapped in the following
        wrappers (from inner to outer, where the outer returns last):
         - `MonitorInfos`
         - `NormalizeReward`

        Parameters
        ----------
        cls : Type[EnvType]
            The type of env to instantiate.
        monitor_deques_size : int, optional
            Size of the monitor deques. Only valid if `monitor_infos=True`.
        normalize_rewards : bool, optional
            Whether to wrap the env in an instance of `NormalizeReward` or not.
        normalization_gamma : float, optional
            Normalization discount factor. Should be the same as the one used by the RL
            agent. Only valid if `normalize_rewards=True`.

        Returns
        -------
        EnvType
            Wrapped instance of the environment.
        """
        env = cls(*env_args, **env_kwargs)
        env = MonitorInfos(env, monitor_deques_size)
        if normalize_rewards:
            env = NormalizeReward(  # type: ignore[assignment]
                env, normalization_gamma, 1e-6
            )
        return env

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.sym_type})"
