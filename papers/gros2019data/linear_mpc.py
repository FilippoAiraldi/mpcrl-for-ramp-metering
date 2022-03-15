# ============================== #
# RL-based Linear EMPC Regulator #
# ============================== #
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import casadi as cs

import sys, os
sys.path.append(os.path.expanduser('~\\Documents\\git\\mpc-tools'))
import mpctools as mpc

from tqdm import tqdm
import matplotlib.pyplot as plt


# TODO:
# to compute the td error, we need L = stage cost + constraint


#################################### DATA #####################################
# simulation params
Tfin = int(5e2)             # length of the simulation
T = 1                       # time step
t = np.arange(0, Tfin, T)   # time vector

# nominal model
An = np.array([[1, 0.25], [0, 1]])
Bn = np.array([[0.0312], [0.25]])

# real model
Ar = np.array([[0.9, 0.35], [0, 1.1]])
Br = np.array([[0.0813], [0.2]])
noise = lambda: np.array([[np.random.uniform(-0.1, 0)], [0]])

# create systems functions (nominal and real)
Nx, Nu = Ar.shape[0], Br.shape[1]


##################################### MPC #####################################
# final weight
Q = np.eye(Nx)
R = np.eye(Nu) * 0.5
K, S_N = mpc.util.dlqr(An, Bn, Q, R)  # final state weight in the optimization

# bounds
lb = {'x': np.array([[0], [-1]]), 'u': -np.ones((Nu, 1))}
ub = {'x': np.array([[1], [1]]), 'u': np.ones((Nu, 1))}

# optimization parameters
N = 10
w = np.full((2, 1), 100)    # relaxation weights
gamma = 0.9                 # discount factor
gamma_vec = np.power(gamma, np.arange(N)).reshape(1, -1)


# stage cost l
def l(x, u):
    return cs.sum1(x**2) + 0.5 * cs.sum1(u**2)


# objective J
def J(x, u, s, V0, f):
    stages = l(x[:, :-1], u) + w.T @ s
    return (V0 +
            0.5 * gamma**N * cs.bilin(S_N, x[:, -1], x[:, -1]) +
            cs.sum2(f.T @ cs.vertcat(x[:, :-1], u)) +
            0.5 * cs.dot(gamma_vec, stages))


# constraint h
def h_lb(x, x_lbar):
    return lb['x'] + x_lbar - x


def h_ub(x, x_ubar):
    return x - ub['x'] - x_ubar


# create MPC
opti = cs.Opti()
# variables
x_var = opti.variable(Nx, N + 1)
u_var = opti.variable(Nu, N)        # control horizon = prediction horizon
s_var = opti.variable(Nx, N)        # slack to relax final state constraint
# parameters
V0_par = opti.parameter(1, 1)       # cost offset
f_par = opti.parameter(Nx + Nu, 1)  # part of the cost
x0_par = opti.parameter(Nx, 1)      # initial conditions
x_lbar_par = opti.parameter(Nx, 1)  # lower constraint tightening
x_ubar_par = opti.parameter(Nx, 1)  # upper constraint tightening
A_par = opti.parameter(Nx, Nx)      # learnt A matrix
B_par = opti.parameter(Nx, Nu)      # learnt B matrix
b_par = opti.parameter(Nx, 1)       # learnt b offset
# objective
opti.minimize(J(x_var, u_var, s_var, V0_par, f_par))
# constraints
opti.subject_to(cs.vec(s_var) >= 0)
for i, (u_lb, u_ub) in enumerate(zip(lb['u'], ub['u'])):
    opti.subject_to(u_var[i, :] >= u_lb)
    opti.subject_to(u_var[i, :] <= u_ub)
for k in range(N):
    opti.subject_to(h_lb(x_var[:, k], x_lbar_par) <= s_var[:, k])
    opti.subject_to(h_ub(x_var[:, k], x_ubar_par) <= s_var[:, k])
opti.subject_to(x_var[:, 0] == x0_par)
for k in range(N):
    opti.subject_to(x_var[:, k + 1] ==
                    A_par @ x_var[:, k] + B_par @ u_var[:, k] + b_par)

# solver
opti.solver('ipopt', {'print_time': False}, {'sb': 'no', 'print_level': 0})


# methods
def sim_traj(x0, u, A, B, b):
    X = [x0]
    for k in range(u.shape[1]):
        X.append(A @ X[-1] + B @ u[:, k].reshape((-1, 1)) + b)
    assert all(o.shape == (2, 1) for o in X)
    return np.hstack(X)


def MPC(opti, k, x0, u0, par, is_Q=False, x_init=None):
    # if Q function, make a copy of opti and add constraint that first action is u0
    if u0.shape[1] == 1:
        u0 = cs.repmat(u0, 1, N)
    if is_Q:
        opti = opti.copy()
        opti.subject_to(u_var[:, 0] == u0[:, 0])
    opti.set_value(x0_par, x0)
    opti.set_value(V0_par, par['V0'])
    opti.set_value(f_par, par['f'])
    opti.set_value(x_lbar_par, par['x_lbar'])
    opti.set_value(x_ubar_par, par['x_ubar'])
    opti.set_value(A_par, par['A'])
    opti.set_value(B_par, par['B'])
    opti.set_value(b_par, par['b'])
    opti.set_initial(s_var, 0)
    opti.set_initial(x_var, sim_traj(x0, u0, par['A'], par['B'], par['b'])
                     if x_init is None else
                     x_init)
    opti.set_initial(u_var, u0)
    try:
        sol = opti.solve()
        get_value = lambda o: sol.value(o)
    except:
        tqdm.write(f'{k}: ' + opti.debug.stats()['return_status'])
        get_value = lambda o: opti.debug.value(o)

    x = get_value(x_var)
    pi = get_value(u_var)
    obj = get_value(opti.f)
    # NOTE: unsure if + or -
    lagr = opti.f + get_value(opti.lam_g).reshape(-1, 1).T @ opti.g

    if is_Q:
        derivatives = {
            'V0': get_value(cs.jacobian(lagr, V0_par)),
            'f': get_value(cs.jacobian(lagr, f_par)).reshape(-1, 1),
            'x_lbar': get_value(cs.jacobian(lagr, x_lbar_par)).reshape(-1, 1),
            'x_ubar': get_value(cs.jacobian(lagr, x_ubar_par)).reshape(-1, 1),
            'A': get_value(cs.jacobian(lagr, A_par)).reshape(Nx, Nx).T,
            'B': get_value(cs.jacobian(lagr, B_par)).reshape(Nx, Nu),
            'b': get_value(cs.jacobian(lagr, b_par)).reshape(-1, 1)
        }
    else:
        derivatives = {}
    return x.reshape(Nx, N + 1), pi.reshape(Nu, N), obj, derivatives


def deepcopy(d):
    return {
        k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()
    }


##################################### RL ######################################
alpha = 1e-6


def L(x, u, x_lbar, x_ubar):
    return l(x, u) + w.T @ cs.fmax(0,
                                   cs.fmax(h_lb(x, x_lbar), h_ub(x, x_ubar)))


def update_params(L, V, Q, Qjac, params):
    td = L + gamma * V - Q
    # TODO: perform update on params, no need to copy, overwrite reference


############################### SIMULATION LOOP ###############################
# initial conditions
params = {
    'V0': 0,
    'f': np.zeros((Nx + Nu, 1)),
    'x_lbar': np.zeros((Nx, 1)),
    'x_ubar': np.zeros((Nx, 1)),
    'A': An,
    'B': Bn,
    'b': np.zeros((Nx, 1))
}
x = np.array([[0.05], [0.2]])
u = np.zeros((Nu, 1))
X, U, Params = [x], [], [deepcopy(params)]

# loop
for k in tqdm(range(Tfin), total=Tfin, desc='Simulation'):

    # # RL update

    # # compute action
    # x_opt, u_opt, Q, params_jac = MPC(opti, k, x, u, params, is_Q=True)
    # _, u_opt, V, _ = MPC(opti, k, x, u_opt, params, x_init=x_opt)
    # u = u_opt[:, 0, None]
    # # u = - K @ x

    # # step system
    # x = Ar @ x + Br @ u + noise()

    # save variables
    X.append(x)
    U.append(u)
    Params.append(deepcopy(params))


################################### PLOTTING ##################################
# generate random data
x = np.hstack(X)
u = np.hstack(U)
A = np.stack([p['A'] for p in Params], axis=-1)
B = np.stack([p['B'] for p in Params], axis=-1)
b = np.stack([p['b'] for p in Params], axis=-1)
f = np.stack([p['f'] for p in Params], axis=-1)
V0 = np.stack([p['V0'] for p in Params], axis=-1)
x_lbar = np.stack([p['x_lbar'] for p in Params], axis=-1)
x_ubar = np.stack([p['x_ubar'] for p in Params], axis=-1)

# create figure
fig = plt.figure(constrained_layout=True, figsize=(10, 8))
figs = fig.subfigures(1, 2, wspace=0.07)

# plot states and control action
figs[0].suptitle('control')
axs = figs[0].subplots(Nx + Nu, 1, sharex=True)
for i in range(Nx):
    axs[i].plot(t, x[i, :-1])
    axs[i].axhline(lb['x'][i], t[0], t[-1], linestyle='--', color='r')
    axs[i].axhline(ub['x'][i], t[0], t[-1], linestyle='--', color='r')
    axs[i].set_ylabel(f'$x_{i}$')
for i in range(Nu):
    axs[i + Nx].plot(t, u[i])
    axs[i + Nx].axhline(lb['u'][i], t[0], t[-1], linestyle='--', color='r')
    axs[i + Nx].axhline(ub['u'][i], t[0], t[-1], linestyle='--', color='r')
    axs[i + Nx].set_ylabel(f'$u_{i}$')
for ax in axs:
    ax.set_xlabel('t')

# plot learning stuff
figs[1].suptitle('learning')
subfigs = figs[1].subfigures(2, 1, height_ratios=[1, 2 / 3])

# plot learnt parameters
subfigs[0].suptitle('parameters')
subfigs[0].set_facecolor('r')
axs = subfigs[0].subplots(3, 2, sharex=True)

# plot TD error and cost
subfigs[1].suptitle('error and cost')
subfigs[1].set_facecolor('g')
axs = subfigs[1].subplots(2, 1, sharex=True)

plt.show()
