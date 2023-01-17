from typing import Literal, Optional, TypeVar

import casadi as cs
from csnlp import MultistartNlp, Nlp
from csnlp.wrappers import Mpc

from util.constants import MpcConstants as MC

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HighwayTrafficMpc(Mpc[SymType]):
    def __init__(self, sym_type: Literal["SX", "MX"] = "SX") -> None:
        starts = MC.multistart
        nlp = (
            Nlp(sym_type=sym_type)
            if starts == 1
            else MultistartNlp(starts=starts, sym_type=sym_type)
        )
        sp = MC.input_spacing
        Np = MC.prediction_horizon * sp
        Nc = MC.control_horizon * sp
        super().__init__(nlp, Np, Nc, sp)
