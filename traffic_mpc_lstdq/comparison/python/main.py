# sourcery skip: move-assign-in-block, use-fstring-for-concatenation, use-named-expression
import numpy as np

import sys, os
sys.path.append(os.path.expanduser('~\\Documents\\git\\metanet'))
import metanet
from metanet import control

from lib import run_sim_with_MPC

import time
import logging
import matplotlib.pyplot as plt


############################# LOGGING and SAVING ##############################


# This is also an exercise to run two simulations at the same time, which only
# share the dynamics
run_name = time.strftime('%Y%m%d_%H%M%S')
save_name = 'result_' + run_name

logging.basicConfig(
    filename='log_' + run_name + '.txt', level=logging.INFO,
    format='[%(levelname)s|%(asctime)s] - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


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
v_free = (102, 130)             # free flow speed (km/h)
rho_crit = (33.5, 27)           # critical capacity (veh/km/lane)

# create all simulations (the wrong nominal one is used to store create the
# mpc with wrong parameters, and to store the corresponding results when run
# with the true dynamics)
net_names = ['Real', 'Nominal']
sims: list[metanet.Simulation] = []
for i in range(len(net_names)):
    # create components
    _N1 = metanet.Node(name='N1')
    _N2 = metanet.Node(name='N2')
    _N3 = metanet.Node(name='N3')
    _L1 = metanet.Link(2, lanes, L, v_free[i], rho_crit[i], a[i], name='L1')
    _L2 = metanet.Link(1, lanes, L, v_free[i], rho_crit[i], a[i], name='L2')
    _O1 = metanet.MainstreamOrigin(name='O1')
    _O2 = metanet.OnRamp(C2, name='O2')
    _D1 = metanet.Destination(name='D1')

    # assemble network
    _net = metanet.Network(name=net_names[i])
    _net.add_nodes(_N1, _N2, _N3)
    _net.add_links((_N1, _L1, _N2), (_N2, _L2, _N3))
    _net.add_origins((_O1, _N1), (_O2, _N2))
    _net.add_destination(_D1, _N3)
    # _net.plot(reverse_x=True, expanded_view=True)

    # create true simulation
    sims.append(metanet.Simulation(_net, T, rho_max, eta, tau, kappa, delta))

sim_true, sim_nom = sims


################################# DISTURBANCES ################################


# # original disturbance
# d1 = np.tile(metanet.util.create_profile(
#     t[:Kstage], [0, .25, 1, 1.25], [1000, 3500, 3500, 1000]), stages).tolist()
# d2 = np.tile(metanet.util.create_profile(
#     t[:Kstage], [.25, .37, .62, .75], [500, 1750, 1750, 500]), stages).tolist()

# # result 1 disturbance - small difference in cost, no slack
# d1 = np.tile(metanet.util.create_profile(
#     t[:Kstage], [0, .3, .95, 1.25], [1000, 3100, 3100, 1000]), stages).tolist()
# d2 = np.tile(metanet.util.create_profile(
#     t[:Kstage], [.15, .32, .57, .75], [500, 1800, 1800, 500]), stages).tolist()

# result 2 disturbance - more difference in cost, some slack
d1 = np.tile(metanet.util.create_profile(
    t[:Kstage], [0, .3, .95, 1.25], [1000, 3150, 3150, 1000]), stages).tolist()
d2 = np.tile(metanet.util.create_profile(
    t[:Kstage], [.15, .32, .57, .75], [500, 1800, 1800, 500]), stages).tolist()


# plt.figure(figsize=(4, 3), constrained_layout=True)
# plt.plot(t, d1, '-', label='$O_1$')
# plt.plot(t, d2, '--', label='$O_2$')
# plt.xlabel('time (h)')
# plt.ylabel('demand (veh/h)')
# plt.xlim(0, Tfin)
# plt.ylim(0, round(max(d1 + d2), -3))
# plt.legend()
# plt.show()


##################################### MPC #####################################


# mpc parameters
Np, Nc, M = 7, 3, 6
cost = lambda s, v, p: (control.TTS(s, v, p) +
                        control.input_variability_penalty(s, v, p, 0.4) +
                        control.slacks_penalty(s, v, p, 10))
max_iter = 3000
w2_constraint = 100
MPCs = []

# create MPC controllers for true and nominal models
for sim in sims:
    plugin_opts = {'expand': True, 'print_time': False,
                   'calc_lam_x': True, 'calc_multipliers': True}
    solver_opts = {'print_level': 0, 'max_iter': max_iter}

    # OPTI mpc
    mpc = control.OptiMPC(
        sim=sim, Np=Np, Nc=Nc, M=M, cost=cost, solver='ipopt',
        plugin_opts=plugin_opts, solver_opts=solver_opts)
    slack1 = mpc.add_slack('1', mpc.vars[f'w_{sim.net.O2}'].shape)
    mpc.opti.subject_to(mpc.vars[f'w_{sim.net.O2}'] - slack1 <= w2_constraint)

    # # NLP mpc
    # mpc = control.NlpSolMPC(
    #     sim=sim, Np=Np, Nc=Nc, M=M, cost=cost, solver='ipopt',
    #     plugin_opts=plugin_opts, solver_opts=solver_opts)
    # slack1 = mpc.add_slack('1', mpc.vars[f'w_{sim.net.O2}'].shape)
    # mpc.add_constraint(-np.inf,
    #                    mpc.vars[f'w_{sim.net.O2}'] - slack1, w2_constraint)

    MPCs.append(mpc)


################################# SIMULATION ##################################


# initial conditions and disturbances
rho0_L1 = np.array([4.98586, 5.10082])
v0_L1 = np.array([100.297, 98.0923])
rho0_L2 = np.array([7.63387])
v0_L2 = np.array([98.4106])
for sim in sims:
    sim.set_init_cond({
        sim.net.L1: (rho0_L1, v0_L1),
        sim.net.L2: (rho0_L2, v0_L2)
    }, {
        sim.net.O1: np.array(0),
        sim.net.O2: np.array(0)
    }, {
        sim.net.O2: np.array(1)
    })
    sim.net.O1.demand = list(d1)  # to be sure it's a copy
    sim.net.O2.demand = list(d2)
    control.steadystate(sim, sim_true=sim_true)


# simulations
exec_time1 = run_sim_with_MPC(sim_true, MPCs[0], t, logger)
exec_time2 = run_sim_with_MPC(sim_nom, MPCs[1], t, logger, sim_true=sim_true)


################################## PLOT/SAVE ##################################


# compute costs
ttss = []
for sim in sims:
    tts = control.TTS(sim, {
        **{f'rho_{l}': np.hstack(l.density) for l in sim.net.links},
        **{f'v_{l}': np.hstack(l.speed) for l in sim.net.links},
        **{f'w_{o}': np.hstack(o.queue) for o in sim.net.origins},
    }, None)
    logger.info(f'{sim.net.name}: TTS = {tts}, J = {sum(sim.objective)}')
    ttss.append(float(tts))

# save results
import platform
data = {
    'platform': str(platform.platform()),
    'algorithms': [mpc.__class__.__name__ for mpc in MPCs],
    'otps': {'Np': Np, 'Nc': Nc, 'M': M, 'max_iter': max_iter},
    'execution_times': (exec_time1, exec_time2),
    'a': a,
    'v_free': v_free,
    'rho_crit': rho_crit,
    'TTS': ttss,
    'J': [sum(sim.objective) for sim in sims],
    'w2_constraint': w2_constraint
}
metanet.io.save_sims(save_name + '.pkl', *sims, **data)
# metanet.io.save_sims(save_name + '.mat', *sims, **data)

# plot results
fig, axs = sim_true.plot(t, sharex=True)
sim_nom.plot(t, fig=fig, axs=axs, linestyle='--', add_labels=False)
plt.show()
