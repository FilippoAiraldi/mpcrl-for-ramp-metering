from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple

import numpy as np
import numpy.typing as npt


class EnvConstants:
    """Constant parameters of the highway traffic network."""

    T: ClassVar[float] = 10 / 3600  # simulation step size (h)
    Tscenario: ClassVar[float] = 2.0  # duration of each demand scenario
    steps: ClassVar[int] = 6  # timesteps to simulate at each env.step calls
    #
    segment_length: ClassVar[float] = 1  # length of links segments (km)
    lanes: ClassVar[int] = 2  # lanes per link (adim)
    origin_capacities: ClassVar[tuple[float, float]] = (
        3500,
        2000,
    )  # on-ramp capacities (veh/h)
    rho_max: ClassVar[float] = 180  # maximum capacity (veh/km/lane)
    a: ClassVar[float] = 1.867  # model parameter (adim)
    v_free: ClassVar[float] = 102  # free flow speed (km/h)
    rho_crit: ClassVar[float] = 33.5  # critical capacity (veh/km/lane)
    tau: ClassVar[float] = 18 / 3600  # model parameter (s)
    kappa: ClassVar[float] = 40  # model parameter (veh/km/lane)
    eta: ClassVar[float] = 60  # model parameter (km^2/lane)
    delta: ClassVar[float] = 0.0122  # merging phenomenum parameter
    #
    ramp_max_queue: ClassVar[dict[str, int]] = {"O2": 50}  # max queues (only ramp O2)
    ramp_min_flow_factor: ClassVar[int] = 10  # min queue on ram
    stage_cost_weights: ClassVar[dict[str, float]] = {  # weight of each contribution
        "tts": 5.0,
        "var": 1.6e3,
        "cvi": 5.0,
    }


# perform some checks
EC = EnvConstants
assert EC.Tscenario / EC.T % EC.steps == 0.0, "Incompatible sim length and step size."
STEPS_PER_SCENARIO = int(EC.Tscenario / EC.T / EC.steps)  # 120 by default


# initialize approximated model parameters (with some mismatches w.r.t. perfect model)
rho_crit_ = EC.rho_crit * 0.7
a_ = EC.a * 1.3
v_free_ = EC.v_free * 1.3


class ParInfo(NamedTuple):
    """Stores information on a parameter in the MPC scheme, in particular
    - value (initial, if learnable)
    - flag indicating whether the parameter is learnable or not.
    - bounds (only if learnable)
    - learning rate multiplier (only if learnable)."""

    value: npt.ArrayLike
    learnable: float = False
    bounds: tuple[npt.ArrayLike, npt.ArrayLike] = (np.nan, np.nan)
    lr_multiplier: float = np.nan


class MpcRlConstants:
    """Constant settings and parameters of the highway traffic MPC-RL controller."""

    prediction_horizon: ClassVar[int] = 4  # prediction horizon \approx 3*L/(M*T*v_avg)
    control_horizon: ClassVar[int] = 3  # control horizon
    #
    structured_multistart: ClassVar[int] = 0  # number of NMPC structured multistarts
    random_multistart: ClassVar[int] = 0  # number of NMPC random multistarts
    solver_opts: ClassVar[dict[str, Any]] = {  # solver options
        "expand": True,
        "print_time": False,
        "bound_consistency": True,
        "clip_inactive_lam": False,
        "calc_lam_x": True,
        "calc_lam_p": False,
        "warn_initial_bounds": False,
        "show_eval_warnings": True,
        # "jit": True,
        # "compiler": "shell",
        # "jit_options": {"flags": ["-O1"], "verbose": True},
        "ipopt": {
            "max_iter": 5e2,
            "tol": 1e-6,
            "barrier_tol_factor": 4,
            # "linear_solver": "pardiso",
            "sb": "yes",
            "print_level": 0,
        },
    }
    #
    normalization: ClassVar[dict[str, float]] = {  # normalization factors for...
        "rho": EC.rho_max,  # density
        "v": v_free_,  # speed
        "w": EC.ramp_max_queue["O2"],  # queue
        "a": EC.origin_capacities[1],  # action
    }
    #
    parameters: ClassVar[MappingProxyType[str, ParInfo]] = MappingProxyType(
        {
            "rho_crit": ParInfo(rho_crit_, False, (10, EC.rho_max * 0.9), 1),
            "a": ParInfo(a_, True, (1.1, 3.0), 1),  # NOTE: must be > 1
            "v_free": ParInfo(v_free_, False, (30, 300), 1),
            "weight_tts": ParInfo(
                EC.stage_cost_weights["tts"] / 5, True, (1e-3, np.inf), 1
            ),
            "weight_var": ParInfo(
                EC.stage_cost_weights["var"] * 100, True, (1e-3, np.inf), 1
            ),
            "weight_slack": ParInfo(
                EC.stage_cost_weights["cvi"], True, (1e-3, np.inf), 1
            ),
            "weight_init_rho": ParInfo(1, True, (-np.inf, np.inf), 1),
            "weight_init_v": ParInfo(1, True, (-np.inf, np.inf), 1),
            "weight_init_w": ParInfo(1, True, (-np.inf, np.inf), 1),
            "weight_stage_rho": ParInfo(1, True, (1e-6, np.inf), 1),
            "weight_stage_v": ParInfo(1, True, (1e-6, np.inf), 1),
            "weight_stage_w": ParInfo(1, True, (1e-6, np.inf), 1),
            "weight_terminal_rho": ParInfo(1, True, (1e-6, np.inf), 1),
            "weight_terminal_v": ParInfo(1, True, (1e-6, np.inf), 1),
            "weight_terminal_w": ParInfo(1, True, (1e-6, np.inf), 1),
        }
    )
