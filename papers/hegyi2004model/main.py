import casadi as cs
import numpy as np

import sys, os
sys.path.append(os.path.expanduser('~\\Documents\\git\\metanet'))
import metanet

from tqdm import tqdm
import time
import matplotlib.pyplot as plt


################################### SCENARIO ##################################


scenario = 2
save_results = False


if scenario == 1:
    print('NO CONTROL')
    Nc = 3
    disable_ramp, disable_vms = True, True
    v_ctrl0 = 9e3
    save_name = 'no_ctrl'
elif scenario == 2:
    print('RAMP ONLY')
    Nc = 3
    disable_ramp, disable_vms = False, True
    v_ctrl0 = 9e3
    save_name = 'ramp_only'
elif scenario == 3:
    print('RAMP-VMS COORDINATION')
    Nc = 5
    disable_ramp, disable_vms = False, False
    v_ctrl0 = 20
    save_name = 'coord'
else:
    raise ValueError('Invalid scenario (only 1, 2, or 3)')

use_mpc = scenario != 1
max_iter, max_cpu = 6000, 30


#################################### MODEL ####################################


# simulation
Tfin = 2.5                      # final simulation time (h)
T = 10 / 3600                   # simulation step size (h)
t = np.arange(0, Tfin + T, T)   # time vector (h)
K = t.size                      # simulation steps

# segments
L = 1                           # length of links (km)
lanes = 2                       # lanes per link (adim)

# on-ramp O2
C2 = 2000                       # on-ramp capacity (veh/h/lane)
max_queue = 100

# Variable speed limits
alpha = 0.1                     # adherence to limit

# model parameters
tau = 18 / 3600                 # model parameter (s)
kappa = 40                      # model parameter (veh/km/lane)
eta = 60                        # model parameter (km^2/lane)
rho_max = 180                   # maximum capacity (veh/km/lane)
delta = 0.0122                  # merging phenomenum parameter
a = 1.867                       # model parameter (adim)
v_free = 102                    # free flow speed (km/h)
rho_crit = 33.5                 # critical capacity (veh/km/lane)

# create components
N1 = metanet.Node(name='N1')
N2 = metanet.Node(name='N2')
N3 = metanet.Node(name='N3')
L1 = metanet.LinkWithVms(4, lanes, L, v_free, rho_crit, a, vms=[2, 3],
                         name='L1')
L2 = metanet.Link(2, lanes, L, v_free, rho_crit, a, name='L2')
O1 = metanet.MainstreamOrigin(name='O1')
O2 = metanet.OnRamp(C2, name='O2')
D1 = metanet.Destination(name='D1')

# assemble network
net = metanet.Network(name='Small example')
net.add_nodes(N1, N2, N3)
net.add_link(N1, L1, N2)
net.add_link(N2, L2, N3)
net.add_origin(O1, N1)
net.add_origin(O2, N2)
net.add_destination(D1, N3)
# net.plot(reverse_x=False, expanded_view=True)

# create simulation
sim = metanet.Simulation(net, T, rho_max, eta, tau, kappa, delta, alpha=alpha)
F = metanet.control.sim2func(sim, out_nonstate=True)


################################# DISTURBANCES ################################


O1.demand = metanet.util.create_profile(t, [2, 2.25], [3500, 1000]).tolist()
O2.demand = metanet.util.create_profile(
    t, [0, .125, .375, 0.5], [500, 1500, 1500, 500]).tolist()

# plt.figure(figsize=(4, 3), constrained_layout=True)
# plt.plot(t, O1.demand, '-', label='$O_1$')
# plt.plot(t, O2.demand, '--', label='$O_2$')
# plt.xlabel('time (h)')
# plt.ylabel('demand (veh/h)')
# plt.xlim(0, Tfin)
# plt.ylim(0, round(max(itertools.chain(O1.demand, O2.demand)), -3))
# plt.legend()
# plt.show()


##################################### MPC #####################################

metanet.control.MPC
# control input multiplier and prediction/control horizon
Np, Nc, M = 7, Nc, 6
cost = lambda s, v, p: metanet.control.TTS_with_input_penalty(
    s, v, p, 0 if disable_ramp else 0.4, 0 if disable_vms else 0.4)

# mpc = metanet.control.MPC(sim=sim, Np=Np, Nc=Nc, M=M, cost=cost,
#                            disable_ramp_metering=disable_ramp,
#                            disable_vms=disable_vms,
#                            plugin_opts={'expand': True, 'print_time': False},
#                            solver_opts={'print_level': 0,
#                                         'max_iter': max_iter,
#                                         'max_cpu_time': max_cpu})
# #   solver='sqpmethod',
# #   plugin_opts={'expand': True, 'print_time': False,
# #   'qpsol': 'qrqp'})
# # expand makes function evaluations faster but requires more memory
# mpc.opti.subject_to(cs.vec(mpc.vars[f'w_{O2}'][:, 1:]) <= 100)
# MPC = mpc.to_func()

opts = {'expand': True, 'print_time': False,
        'ipopt': {
            'print_level': 0, 'max_iter': max_iter, 'max_cpu_time': max_cpu}}
mpc = metanet.control.NlpSolver(sim=sim, Np=Np, Nc=Nc, M=M, cost=cost,
                                disable_ramp_metering=disable_ramp,
                                disable_vms=disable_vms,
                                solver='ipopt', solver_opts=opts)

mpc.add_constraint(0, mpc.vars[f'w_{O2}'], 100)
MPC = mpc.to_func()


################################# SIMULATION ##################################


# initial conditions
rho0_L1 = np.array([21.864, 21.9439, 22.3086, 23.8106])
v0_L1 = np.array([80.0403, 79.7487, 78.4452, 73.4967])
rho0_L2 = np.array([29.3076, 30.1582])
v0_L2 = np.array([68.2416, 66.317])
sim.set_init_cond({L1: (rho0_L1, v0_L1), L2: (rho0_L2, v0_L2)},
                  {O1: np.array(0), O2: np.array(0)})

# simulate MPC
start_time = time.time()
metanet.control.run_sim_with_MPC(sim, mpc, K, use_tqdm=True)
execution_time = time.time() - start_time


# # initialize last solution
# vars_last = {
#     **{f'w_{o}': cs.repmat(o.queue[0], 1, M * Np + 1) for o in net.origins},
#     **{f'rho_{l}': cs.repmat(l.density[0], 1, M * Np + 1) for l in net.links},
#     **{f'v_{l}': cs.repmat(l.speed[0], 1, M * Np + 1) for l in net.links},
#     **{f'r_{o}': np.ones((1, Nc)) for o, _ in sim.net.onramps},
#     **{f'v_ctrl_{l}': np.full((l.nb_vms, Nc), v_ctrl0)
#         for l, _ in sim.net.links_with_vms}
# }

# # simulation main loop
# start_time = time.time()
# for k in tqdm(range(K), total=K):
#     # compute control action
#     if k % M == 0 and use_mpc:
#         # get future demands
#         dist = {}
#         for origin in net.origins:
#             d = origin.demand[k:k + M * Np]
#             dist[f'd_{origin}'] = np.pad(d, (0, M * Np - len(d)),
#                                          mode='edge').reshape(1, -1)
#         # run MPC
#         vars_init = {
#             var: metanet.util.shift(val, axis=2)
#             for var, val in vars_last.items()
#         }
#         pars_val = {
#             **dist,
#             **{f'w0_{o}': o.queue[k] for o in net.origins},
#             **{f'rho0_{l}': l.density[k] for l in net.links},
#             **{f'v0_{l}': l.speed[k] for l in net.links},
#             **{f'r_{o}_last': vars_last[f'r_{o}'][0, 0]
#                 for o, _ in sim.net.onramps},
#             **{f'v_ctrl_{l}_last': vars_last[f'v_ctrl_{l}'][:, 0]
#                 for l, _ in sim.net.links_with_vms},
#         }
#         vars_last, info = MPC(vars_init, pars_val)
#         if 'error' in info:
#             tqdm.write(f'{k:{len(str(K))}}/{t[k]:.2f}:' + info['error'] + '.')

#     # set onramp metering rate and vms speed control
#     for o, _ in net.onramps:
#         o.rate[k] = vars_last[f'r_{o}'][0, 0]
#     for l, _ in net.links_with_vms:
#         l.v_ctrl[k] = vars_last[f'v_ctrl_{l}'][:, 0].reshape((l.nb_vms, 1))

#     # step simulate
#     (O1.flow[k], O1.queue[k + 1],
#      O2.flow[k], O2.queue[k + 1],
#      L1.flow[k], L1.density[k + 1], L1.speed[k + 1],
#      L2.flow[k], L2.density[k + 1], L2.speed[k + 1]
#      ) = F(O1.queue[k],                    # -
#            O2.queue[k],                    # | states
#            L1.density[k], L1.speed[k],     # |
#            L2.density[k], L2.speed[k],     # -
#            O2.rate[k], L1.v_ctrl[k],       # inputs
#            O1.demand[k], O2.demand[k])     # disturbances

# execution_time = (time.time() - start_time)


################################## PLOT/SAVE ##################################


tts = metanet.control.TTS(sim, {
    **{f'rho_{l}': np.hstack(l.density) for l in sim.net.links},
    **{f'v_{l}': np.hstack(l.speed) for l in sim.net.links},
    **{f'w_{o}': np.hstack(o.queue) for o in sim.net.origins},
}, None)
print(f'TTS = {tts}')

if save_results:
    import platform
    data = {
        'execution_time': float(execution_time),
        'platform': str(platform.platform()),
        'algorithm': mpc.__class__.__name__,
        'TTS': float(tts)
    }
    metanet.io.save_sim(save_name + '.pkl', sim, **data)
    metanet.io.save_sim(save_name + '.mat', sim, **data)

sim.plot(t, sharex=True)
plt.show()
# fig.savefig('test.eps')
