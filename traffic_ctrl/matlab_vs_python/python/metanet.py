import casadi as cs
import numpy as np
from dataclasses import dataclass

from typing import List


def create_profile(t: List[int], x: List[int], y: List[int]) -> np.ndarray:
    '''
    Creates a piece-wise linear profile along time vector t, passing through
    all x and y points.

    Parameters
    ----------
        t : {list, array}
            1D array or list representing the (monotonically increasing) time
            vector.
        x : {list, array}
            1D array or list of x-coordinates where the piecewise profile
            passes through.
        y : {list, array}
            1D array or list of y-coordinates where the piecewise profile
            passes through. Must have same length as x.

    Returns
    -------
        profile : array
            the piecewise linear function, of the same size of t.

    Raises
    ------
        ValueError : length mismatch
            If x and y don't share the same length
    '''

    if len(x) != len(y):
        raise ValueError('length mismatch between \'x\' and \'y\' '
                         f'({len(x)} vs {len(y)})')

    # convert x from time to indices
    t = np.array(t)
    x = list(map(lambda p: np.argmin(np.abs(t - p)), x))

    # add first and last timestep
    if x[0] != 0:
        x = [0, *x, t.size - 1]
        y = [y[0], *y, y[-1]]
    else:
        x = [*x, t.size - 1]
        y = [*y, y[-1]]

    # create profile of piecewise affine functions (i.e., lines)
    profile = np.zeros_like(t)
    for i in range(len(x) - 1):
        m = (y[i + 1] - y[i]) / (t[x[i + 1]] - t[x[i]])
        q = y[i] - m * t[x[i]]
        profile[x[i]:x[i + 1]] = m * t[x[i]:x[i + 1]] + q

    # take care of last point
    profile[-1] = profile[-2]
    return profile


def shift(x: cs.SX, n: int = 1, axis: int = 1) -> cs.SX:
    '''
    Shifts the array along the axis and pads with last value.
        [1,2,3,4]    --- shift n=2 --> [3,4,4,4]

    Parameters
    ----------
        n : int, optional
            Size of shift. Can also be negative. Defaults to 1.

        axis : int, {1, 2}, optional
            Axis along which to shift.

    Raises
    ------
        ValueError : invalid axis
            If the axis is invalid.
    '''

    if axis not in (1, 2):
        raise ValueError(f'Invalid axis; expected 1 or 2, got {axis} instead.')

    if n == 0:
        return x

    if n > 0:
        if axis == 1:
            return cs.vertcat(x[n:, :], *([x[-1, :]] * n))
        return cs.horzcat(x[:, n:], *([x[:, -1]] * n))

    if axis == 1:
        return cs.vertcat(*([x[0, :]] * (-n)), x[:n, :])
    return cs.horzcat(*([x[:, 0]] * (-n)), x[:, :n])


def build_input(x0, x_last, M):
    x_last = shift(x_last, n=M, axis=2)
    x_last[:, 0] = x0
    return x_last


@dataclass
class Config:
    L: float
    lanes: float
    v_free: float
    rho_crit: float
    rho_max: float

    a: float
    tau: float
    eta: float
    kappa: float
    delta: float

    C2: float

    T: float


def Veq(rho, v_free, a, rho_crit):
    return v_free * cs.exp((-1 / a) * cs.power(rho / rho_crit, a))


def f(w, rho, v,  # states
      r2,         # inputs
      d,          # disturbances
      config: Config):

    # unpack some things
    T = config.T
    L = config.L
    lanes = config.lanes
    v_free = config.v_free
    rho_crit = config.rho_crit
    rho_max = config.rho_max
    a = config.a
    tau = config.tau
    eta = config.eta
    kappa = config.kappa
    delta = config.delta

    ########################## ORIGINS ##########################
    # compute flow at mainstream origin O1
    V_rho_crit = Veq(rho_crit, v_free, a, rho_crit)
    v_lim1 = v[0]  # cs.fmin(v_ctrl1, v1)
    q_cap1 = lanes * V_rho_crit * rho_crit
    q_speed1 = (lanes * v_lim1 * rho_crit *
                cs.power(-a * cs.log(v_lim1 / v_free), 1 / a))
    q_lim1 = cs.if_else(v_lim1 < V_rho_crit, q_speed1, q_cap1)
    q_O1 = cs.fmin(d[0] + w[0] / T, q_lim1)

    # compute flow at onramp origin O2
    q_O2 = cs.fmin(d[1] + w[1] / T, config.C2 *
                   cs.fmin(r2, (rho_max - rho[2]) / (rho_max - rho_crit)))

    # step queue at origins O1 and O2
    q_o = cs.vertcat(q_O1, q_O2)
    w_o_next = w + T * (d - q_o)

    ########################## BOUNDARIES ##########################
    # compute link flows
    q = lanes * rho * v

    # compute upstream flow
    q_up = cs.vertcat(q_O1, q[0], q[1] + q_O2)

    # compute upstream speed
    v_up = cs.vertcat(v[0], v[0], v[1])

    # compute downstream density
    rho_down = cs.vertcat(rho[1], rho[2], cs.fmin(rho[2], rho_crit))

    ########################## LINKS ##########################
    # step link densities
    rho_next = rho + (T / (L * lanes)) * (q_up - q)

    # compute V
    V = Veq(rho, v_free, a, rho_crit)

    # step the speeds of the links
    v_next = (v
              + T / tau * (V - v)
              + T / L * v * (v_up - v)
              - eta * T / tau / L * (rho_down - rho) / (rho + kappa))
    v_next[2] -= delta * T / L / lanes * q_O2 * v[2] / (rho[2] + kappa)

    ########################## OUTPUTS ##########################
    # return q_o, w_o_next, q, rho_next, v_next
    return (cs.fmax(q_o, 0), cs.fmax(w_o_next, 0),
            cs.fmax(q, 0), cs.fmax(rho_next, 0), cs.fmax(v_next, 0))


def f2casadiF(config: Config):
    w = cs.SX.sym('w', 2, 1)
    rho = cs.SX.sym('rho', 3, 1)
    v = cs.SX.sym('v', 3, 1)
    r = cs.SX.sym('r', 1, 1)
    d = cs.SX.sym('d', 2, 1)

    args = [w, rho, v, r, d]
    outs = f(*args, config)
    return cs.Function('F', args, outs,
                       ['w', 'rho', 'v', 'r', 'd'],
                       ['q_o', 'w_o_next', 'q', 'rho_next', 'v_next'])


def TTS(w, rho, T, L, lanes):
    return T * cs.sum2(cs.sum1(w) + cs.sum1(rho * L * lanes))


def input_variability_penalty(r_last, r):
    return cs.sum2(cs.diff(cs.horzcat(r_last, r), 1, 1)**2)
