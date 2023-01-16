"""
This file contains all the quantities that are constant, i.e., used in the definition
and characterization of the network architecture, or that are not subject to fine-tuning
for the learning process, i.e., mpc horizons and solver options.
"""

from typing import Any, ClassVar


class EnvConstants:
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
    demands_type: ClassVar[str] = "random"  # (or "constant") type of demand generation
    w_max: ClassVar[dict[str, int]] = {"O2": 50}  # max queue on ramp O2
    stage_cost_weights: ClassVar[dict[str, float]] = {  # weight of each contribution
        "tts": 1.0,
        "var": 0.04,
        "cvi": 10.0,
    }


