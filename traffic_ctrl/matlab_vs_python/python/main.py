import casadi as cs
import numpy as np
import metanet

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


#################################### MODEL ####################################


# simulation
Tstage = 1.5                    # simulation time per stage (h)
stages = 1                      # number of repetitions basically
Tfin = Tstage * stages          # final simulation time (h)
T = 10 / 3600                   # simulation step size (h)
t = np.arange(0, Tfin, T)       # time vector (h)
K = t.size                      # simulation steps
Kstage = K // stages            # simulation steps per stage

# segments
L = 1                           # length of links (km)
lanes = 2                       # lanes per link (adim)

# on-ramp O2
C2 = 2000                       # on-ramp capacity (veh/h/lane)
max_queue = 100

# model parameters
tau = 18 / 3600                 # model parameter (s)
kappa = 40                      # model parameter (veh/km/lane)
eta = 60                        # model parameter (km^2/lane)
rho_max = 180                   # maximum capacity (veh/km/lane)
delta = 0.0122                  # merging phenomenum parameter

# true and wrong model parameters
a = (1.867, 2.111)              # model parameter (adim)
v_free = (102, 120)             # free flow speed (km/h)
rho_crit = (33.5, 27)           # critical capacity (veh/km/lane)

# build configuration
config = metanet.Config(
    L, lanes, v_free[0],
    rho_crit[0],
    rho_max, a[0],
    tau, eta, kappa, delta, C2, T)


################################# DISTURBANCES ################################


d1 = np.tile(metanet.create_profile(
    t[:Kstage], [0, .25, 1, 1.25], [1000, 3500, 3500, 1000]), stages).tolist()
d2 = np.tile(metanet.create_profile(
    t[:Kstage], [.25, .37, .62, .75], [500, 1750, 1750, 500]), stages).tolist()
D = np.vstack((d1, d2))


##################################### MPC #####################################


# mpc parameters
Np, Nc, M = 7, 3, 6
opti = cs.Opti()

# create variables
w_opti = opti.variable(2, M * Np + 1)   # queues
rho_opti = opti.variable(3, M * Np + 1)  # densities
v_opti = opti.variable(3, M * Np + 1)   # speeds
r_opti = opti.variable(1, Nc)           # ramp metering
d_opti = opti.parameter(2, M * Np)      # demands
w0_opti = opti.parameter(2, 1)
rho0_opti = opti.parameter(3, 1)
v0_opti = opti.parameter(3, 1)
r_last_opti = opti.parameter(1, 1)
# w_opti = cs.SX.sym('w', 2, M * Np + 1)
# rho_opti = cs.SX.sym('rho', 3, M * Np + 1)
# v_opti = cs.SX.sym('v', 3, M * Np + 1)
# r_opti = cs.SX.sym('r', 1, Nc)
# d_opti = cs.SX.sym('d', 2, M * Np)
# w0_opti = cs.SX.sym('w0', 2, 1)
# rho0_opti = cs.SX.sym('rho0', 3, 1)
# v0_opti = cs.SX.sym('v0', 3, 1)
# r_last_opti = cs.SX.sym('r_last', 1, 1)

# create casadi function
F = metanet.f2casadiF(config)

# cost to minimize
cost = (metanet.TTS(w_opti, rho_opti, T, L, lanes) +
        0.4 * metanet.input_variability_penalty(r_last_opti, r_opti))
opti.minimize(cost)

# constraints on domains
opti.bounded(0.2, r_opti, 1)
opti.subject_to(cs.vec(w_opti) >= 0)
opti.subject_to(cs.vec(rho_opti) >= 0)
opti.subject_to(cs.vec(v_opti) >= 0)

# constraints on initial conditions
opti.subject_to(w_opti[:, 0] == w0_opti)
opti.subject_to(v_opti[:, 0] == v0_opti)
opti.subject_to(rho_opti[:, 0] == rho0_opti)

# constraints on state evolution
r_exp = cs.horzcat(*[cs.repmat(r_opti[:, i], 1, M) for i in range(Nc)],
                   cs.repmat(r_opti[:, -1], 1, M * (Np - Nc)))
for k in range(M * Np):
    _, w_next, _, rho_next, v_next = F(w_opti[:, k], rho_opti[:, k],
                                       v_opti[:, k], r_exp[:, k], d_opti[:, k])
    opti.subject_to(w_opti[:, k + 1] == w_next)
    opti.subject_to(rho_opti[:, k + 1] == rho_next)
    opti.subject_to(v_opti[:, k + 1] == v_next)

# set solver for opti
plugin_opts = {'expand': True, 'print_time': False}
solver_opts = {'print_level': 0, 'max_iter': 6000,
               'max_cpu_time': 20}
opti.solver('ipopt', plugin_opts, solver_opts)
# plugin_opts = {'qpsol': 'qrqp', 'expand': True, 'print_time': False}
# opti.solver('sqpmethod', plugin_opts)


def MCP(d, w0, rho0, v0, r0, w_last, rho_last, v_last, r_last):
    # set parameters
    opti.set_value(d_opti, d)
    opti.set_value(w0_opti, w0)
    opti.set_value(rho0_opti, rho0)
    opti.set_value(v0_opti, v0)
    opti.set_value(r_last_opti, r0)

    # warm start
    opti.set_initial(w_opti, w_last)
    opti.set_initial(rho_opti, rho_last)
    opti.set_initial(v_opti, v_last)
    opti.set_initial(r_opti, r_last)

    try:
        sol = opti.solve()
        info = {}
        get_value = lambda o: sol.value(o)
    except Exception as ex1:
        try:
            info = {'error': opti.debug.stats()['return_status']}
            get_value = lambda o: opti.debug.value(o)
        except Exception as ex2:
            raise RuntimeError(
                'error during handling of first '
                f'exception.\nEx. 1: {ex1}\nEx. 2: {ex2}') from ex2

    return (get_value(opti.f),
            get_value(w_opti).reshape(w_opti.shape),
            get_value(rho_opti).reshape(rho_opti.shape),
            get_value(v_opti).reshape(v_opti.shape),
            get_value(r_opti).reshape(r_opti.shape)), info


################################# SIMULATION ##################################


# create containers
w = np.full((w_opti.shape[0], K + 1), np.nan)
q_o = np.full((w_opti.shape[0], K), np.nan)
r = np.full((1, K), np.nan)
q = np.full((v_opti.shape[0], K), np.nan)
rho = np.full((rho_opti.shape[0], K + 1), np.nan)
v = np.full((v_opti.shape[0], K + 1), np.nan)

# initial conditions
w[:, 0] = np.zeros(w0_opti.shape).flatten()
rho[:, 0] = np.array([4.98586, 5.10082, 7.63387])
v[:, 0] = np.array([100.297, 98.0923, 98.4106])
r[:, 0] = np.array(0.5).reshape(1, 1)
w_last = np.tile(w[:, 0, None], M * Np + 1)
rho_last = np.tile(rho[:, 0, None], M * Np + 1)
v_last = np.tile(v[:, 0, None], M * Np + 1)
r_last = np.tile(r[:, 0, None], Nc)


# override rate from a matlab file
use_matlab_rate = False
if use_matlab_rate:
    from scipy import io
    r_matlab = io.loadmat('rate_matlab.mat', simplify_cells=True)['r']


# loop
for k in tqdm(range(K), total=K):
    if k % M == 0 and not use_matlab_rate:
        # predict disturbances
        d = D[:, k:k + M * Np]
        d = np.pad(d, ((0, 0), (0, M * Np - d.shape[1])), mode='edge')

        # run MPC
        (_, w_last, rho_last, v_last, r_last), info = MCP(
            d, w[:, k], rho[:, k], v[:, k], r_last[:, 0],
            metanet.build_input(w[:, k], w_last, M),
            metanet.build_input(rho[:, k], rho_last, M),
            metanet.build_input(v[:, k], v_last, M),
            metanet.shift(r_last, axis=2))
        if 'error' in info:
            tqdm.write(f'{k:{len(str(K))}}/{k / K * 100:2.1f}%: '
                       + info['error'] + '.')

    if use_matlab_rate:
        r[:, k] = r_matlab[k]
    else:
        # assign input
        r[:, k] = r_last[:, 0]

    # step sim
    q_o[:, k, None], w[:, k + 1, None], q[:, k, None], rho[:, k + 1, None], v[:,
                                                                              k + 1, None] = F(w[:, k], rho[:, k], v[:, k], r[:, k], D[:, k, None])


################################### PLOTTING ##################################


save_rate = True
if save_rate:
    from scipy import io
    io.savemat('rate_python.mat', {'r': r.flatten()})


# compute cost
tts = metanet.TTS(w, rho, T, L, lanes)
print(f'TTS = {tts}')

# create figure and axis
fig = plt.figure(figsize=(10, 7), constrained_layout=True)
gs = GridSpec(4, 2, figure=fig)
axs = [fig.add_subplot(gs[i, j])
       for i in range(gs.nrows) for j in range(gs.ncols)]
axs = np.array(axs).reshape(gs.nrows, gs.ncols)


# plot link data
for i in range(rho.shape[0]):
    axs[0, 0].plot(t, v[i, :-1], color=f'C{i}', label=f'$v_{{L_{i + 1}}}$')
    axs[0, 1].plot(t, q[i], color=f'C{i}', label=f'$q_{{L_{i + 1}}}$')
    axs[1, 0].plot(t, rho[i, : -1],
                   color=f'C{i}', label=f'$\\rho_{{L_{i + 1}}}$')

# plot origin data
for i in range(w.shape[0]):
    axs[2, 0].plot(t, D[i], color=f'C{i}', label=f'$d_{{O_{i + 1}}}$')
    axs[2, 1].plot(t, w[i, : -1],
                   color=f'C{i}', label=f'$\\omega_{{O_{i + 1}}}$')
    axs[3, 0].plot(t, q_o[i], color=f'C{i}', label=f'$q_{{O_{i + 1}}}$')
axs[3, 1].plot(t, r.flatten(), color='C1', label='$r_{O_2}$')

# set labels
axs[0, 0].set_ylabel('speed (km/h)')
axs[0, 1].set_ylabel('flow (veh/h)')
axs[1, 0].set_ylabel('density (veh/km)')
axs[1, 1].set_axis_off()
axs[2, 0].set_ylabel('origin demand (veh/h)')
axs[2, 1].set_ylabel('queue length (veh)')
axs[3, 0].set_ylabel('origin flow (veh/h)')
axs[3, 1].set_ylabel('metering rate')
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        if (i, j) != (1, 1):
            axs[i, j].sharex(axs[0, 0])
            axs[i, j].set_xlabel('time (h)')
            axs[i, j].set_xlim(0, t[-1])
            axs[i, j].set_ylim(0, axs[i, j].get_ylim()[1])
            axs[i, j].legend()

plt.show()
