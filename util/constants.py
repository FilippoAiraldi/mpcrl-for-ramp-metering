"""
This file contains all the quantities that are constant, i.e., used in the definition
and characterization of the network architecture, or that are not subject to fine-tuning
for the learning process, i.e., mpc horizons and solver options.
"""

from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple

import numpy as np
import numpy.typing as npt


class EnvConstants:
    """Constant parameters of the highway traffic network."""

    T: ClassVar[float] = 10 / 3600  # simulation step size (h)
    Tfin: ClassVar[float] = 2.0  # duration of each demand scenario
    steps: ClassVar[int] = 6  # timesteps to simulate at each env.step calls
    #
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
    #
    demands_type: ClassVar[str] = "constant"  # (or "random") type of demand generation
    #
    ramp_max_queue: ClassVar[dict[str, int]] = {"O2": 50}  # max queue on ramp O2
    ramp_min_flow_factor: ClassVar[int] = 10  # min queue on ram
    stage_cost_weights: ClassVar[dict[str, float]] = {  # weight of each contribution
        "tts": 1.0,
        "var": 4e-2,
        "cvi": 10.0,
        "erm": 10.0,
    }
    erm_robustness: ClassVar[float] = 0.2  # in [0, 1]


EC = EnvConstants
assert EC.Tfin / EC.T % EC.steps == 0.0, "Incompatible simulation length and step size."


class ParInfo(NamedTuple):
    """Stores info of one of the MPC parameters, in particular
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
    multistart: ClassVar[int] = 1  # number of NMPC multistarts
    solver_opts: ClassVar[dict[str, Any]] = {  # solver options
        "expand": True,
        "print_time": False,
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
    parameters: ClassVar[MappingProxyType[str, ParInfo]] = MappingProxyType(
        {
            "rho_crit": ParInfo(EC.rho_crit * 0.7, True, (10, EC.rho_max * 0.9), 1),
            "rho_crit_stage": ParInfo(
                EC.rho_crit * 0.7, True, (10, EC.rho_max * 0.9), 1
            ),
            "rho_crit_terminal": ParInfo(
                EC.rho_crit * 0.7, True, (10, EC.rho_max * 0.9), 1
            ),
            "a": ParInfo(EC.a * 1.3, True, (1.0, 3.0), 1),
            "v_free": ParInfo(EC.v_free * 1.3, True, (30, 250), 1),
            "v_free_stage": ParInfo(EC.v_free * 1.3, True, (30, 250), 1),
            "v_free_terminal": ParInfo(EC.v_free * 1.3, True, (30, 250), 1),
            "weight_tts": ParInfo(
                EC.stage_cost_weights["tts"], True, (1e-3, np.inf), 1
            ),
            "weight_var": ParInfo(
                EC.stage_cost_weights["var"], True, (1e-8, np.inf), 1
            ),
            "weight_slack": ParInfo(
                EC.stage_cost_weights["cvi"], True, (1e-3, np.inf), 1
            ),
            "weight_slack_terminal": ParInfo(
                EC.stage_cost_weights["cvi"], True, (1e-3, np.inf), 1
            ),
            "weight_init_rho": ParInfo(1e-2, True, (-np.inf, np.inf), 1),
            "weight_init_v": ParInfo(1e-3, True, (-np.inf, np.inf), 1),
            "weight_init_w": ParInfo(1e-1, True, (-np.inf, np.inf), 1),
            "weight_stage_v": ParInfo(1e-3, True, (1e-5, np.inf), 1),
            "weight_stage_rho_scale": ParInfo(1e-2, True, (1e-5, np.inf), 1),
            "weight_stage_rho_threshold": ParInfo(1e-1, True, (1e-5, np.inf), 1),
            "weight_terminal_v": ParInfo(1e-3, True, (1e-5, np.inf), 1),
            "weight_terminal_rho_scale": ParInfo(1e-2, True, (1e-5, np.inf), 1),
            "weight_terminal_rho_threshold": ParInfo(1e-1, True, (1e-5, np.inf), 1),
        }
    )
