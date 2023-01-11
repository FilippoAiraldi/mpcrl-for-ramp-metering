from typing import Any, ClassVar, Literal, Optional, SupportsFloat

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import sym_metanet
from csnlp.util.io import SupportsDeepcopyAndPickle
from gymnasium.spaces import Box

from metanet.demands import create_demands
from metanet.network import get_network
from metanet.steady_state import get_steady_state


class Constants:
    """Constant parameters of the highway traffic network."""

    T: ClassVar[float] = 10 / 3600  # simulation step size (h)
    segment_length: ClassVar[float] = 1  # length of links segments (km)
    lanes: ClassVar[int] = 2  # lanes per link (adim)
    origin_capacities: ClassVar[tuple[float, float]] = (
        3500,
        2000,
    )  # on-ramp capacities (veh/h/lane)
    rho_max: ClassVar[float] = 180  # maximum capacity (veh/km/lane)
    a: ClassVar[float] = 1.867  # model parameter (adim)
    v_free: ClassVar[float] = 102  # free flow speed (km/h)
    rho_crit: ClassVar[float] = 33.5  # critical capacity (veh/km/lane)
    tau: ClassVar[float] = 18 / 3600  # model parameter (s)
    kappa: ClassVar[float] = 40  # model parameter (veh/km/lane)
    eta: ClassVar[float] = 60  # model parameter (km^2/lane)
    delta: ClassVar[float] = 0.0122  # merging phenomenum parameter


class HighwayTrafficEnv(
    gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    SupportsDeepcopyAndPickle,
):
    __slots__ = (
        "network",
        "realpars",
        "dynamics",
        "n_scenarios",
        "time",
        "state",
        "demands",
        "reward_range",
        "observation_space",
        "action_space",
        "_np_random",
    )

    def __init__(
        self,
        n_scenarios: int,
        scenario_duration: float,
        sym_type: Literal["SX", "MX"],
    ) -> None:
        gym.Env.__init__(self)
        SupportsDeepcopyAndPickle.__init__(self)
        self.n_scenarios = n_scenarios
        self.time = np.arange(0.0, scenario_duration, Constants.T)

        # create dynamics
        self.network, sympars = get_network(
            segment_length=Constants.segment_length,
            lanes=Constants.lanes,
            origin_capacities=Constants.origin_capacities,
            rho_max=Constants.rho_max,
            sym_type=sym_type,
        )
        self.network.step(
            init_conditions={self.network.origins_by_name["O1"]: {"r": 1}},
            T=Constants.T,
            tau=Constants.tau,
            eta=Constants.eta,
            kappa=Constants.kappa,
            delta=Constants.delta,
        )
        self.dynamics: cs.Function = sym_metanet.engine.to_function(
            net=self.network,
            T=Constants.T,
            parameters=sympars,
            more_out=True,
            force_positive_speed=True,
            compact=2,
        )
        self.realpars = np.asarray([getattr(Constants, n) for n in sympars])
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

        # set reward/cost ranges, and observation and action spaces
        self.reward_range = (0.0, float("inf"))
        self.observation_space = Box(0.0, np.inf, (self.ns,), np.float64)
        na = self.na
        self.action_space = Box(0.0, np.inf, () if na == 1 else (na,), np.float64)

    @property
    def ns(self) -> int:
        """Gets the number of states/observations in the environment."""
        return self.dynamics.size1_in(0)

    @property
    def na(self) -> int:
        """Gets the number of actions in the environment."""
        return self.dynamics.size1_in(1)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        if options is None:
            options = {}

        # reset seeds
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        # create demands
        self.demands = create_demands(
            self.time,
            self.n_scenarios,
            kind=options.get("demands_kind", "constant"),
            noise=options.get("demands_noise", (100.0, 100.0, 2.5)),
            np_random=self.np_random,
        )

        # compute initial state
        x0: np.ndarray = options.get(
            "steady_state_x0", np.asarray([10, 10, 10, 100, 100, 100, 0, 0])
        )
        u = options.get("steady_state_u", 1e3)  # fully open on-ramp O2
        d = self.demands[0]
        p = self.realpars
        f = lambda x: self.dynamics(x, u, d, p)[0].full().reshape(-1)
        state, _, _ = get_steady_state(
            f=f,
            x0=x0,
            tol=options.get("steady_state_tol", 1e-3),
            maxiter=options.get("steady_state_maxiter", 500),
        )
        assert self.observation_space.contains(state), "Invalid reset state."
        self.state = state
        return state, {}
