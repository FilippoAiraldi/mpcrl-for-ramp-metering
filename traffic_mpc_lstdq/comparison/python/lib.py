import casadi as cs
import numpy as np
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from typing import Union, Callable, Dict, Any

import metanet
from metanet import Simulation, Origin
from metanet.ctrl import OptiMPC, NlpSolMPC
from metanet.ctrl.util import sim2func


def fmt(k: int, K: int, t: float, msg: str = None) -> str:
    '''Formats a message for the logger'''
    msg_ = f'[{k:{len(str(K))}}|{k / K * 100:4.1f}%|{timedelta(hours=t)}]'
    if msg is not None:
        msg_ += f' - {msg}'
    return msg_


def run_sim_with_MPC(
    sim: Simulation,
    MPC: Union[OptiMPC, NlpSolMPC],
    t: np.ndarray,
    logger: logging.Logger,
    sim_true: Simulation = None,
    demands_known: bool = True,
    *cbs: Callable[[int, Simulation, Dict[str, float], Dict[str, Any]], None]
) -> float:
    '''
    Automatically run the simulation with an MPC (not necessarily created with 
    the same sim). Be sure to set the initial conditions before calling this 
    method.

    Parameters
    ----------
        sim : metanet.Simulation
            Simulation to run (will contain the results).

        mpc : metanet.control.MPC or metanet.control.NlpSolver 
            MPC controller to run along with the simulation.

        t : np.ndarray
            Simulation time vector.

        logger : logging.Logger
            Logger where to log informations and errors to.

        sim_true : metanet.Simulation, optional
            If provided, the true system dynamics will be taken from this 
            simulation, while results will still be saved in 'sim'. If not 
            provided, it is assumed that 'sim' is governed by the true 
            dynamics. Defaults to None.

        demands_known : bool, optional
            Whether the future demands are known, or only the current value can
            be used. Defaults to True.

        cbs : Callable[iter, sim, vars, info]
            Callbacks called at the end of each iteration.
            NB: should also pass the dict of parameters to update their values

    Returns
    -------
        exec time : float
            Execution time in seconds
    '''

    # create functions
    F = sim2func(sim if sim_true is None else sim_true,
                 out_nonstate=True, out_nonneg=True)

    # save some stuff
    M, Np, Nc = MPC.M, MPC.Np, MPC.Nc
    name = sim.net.name
    origins = list(sim.net.origins.keys())
    onramps = list(map(lambda o: o[0], sim.net.onramps))
    links = list(sim.net.links.keys())
    links_vms = list(map(lambda o: o[0], sim.net.links_with_vms))

    # create a buffer where to save other dynamic data (slacks)
    for slack in MPC.slacks:
        sim.slacks[slack] = []

    # initialize true and nominal last solutions
    vars_last = {}
    for o in origins:
        vars_last[f'w_{o}'] = cs.repmat(o.queue[0], 1, M * Np + 1)
    for l in links:
        vars_last[f'rho_{l}'] = cs.repmat(l.density[0], 1, M * Np + 1)
        vars_last[f'v_{l}'] = cs.repmat(l.speed[0], 1, M * Np + 1)
    for onramp in onramps:
        vars_last[f'r_{onramp}'] = cs.repmat(onramp.rate[0], 1, Nc)
    for link in links_vms:
        vars_last[f'v_ctrl_{l}'] = cs.repmat(link.v_ctrl[0], 1, Nc)
    for slackname, slack in MPC.slacks.items():
        vars_last[slackname] = np.zeros(slack.shape)

    # initialize function to retrieve demands
    if demands_known:
        def get_demand(orig: Origin, k: int) -> np.ndarray:
            d = orig.demand[k:k + M * Np]
            return np.pad(d, (0, M * Np - len(d)), mode='edge').reshape(1, -1)
    else:
        get_demand = lambda orig, k: np.tile(orig.demand[k], (1, M * Np))

    # initialize function to shift arrays in time
    shift = lambda x, n: metanet.util.shift(x, n=n, axis=2)

    # get total amount of simulation steps
    K = t.size

    # simulation main loop
    with logging_redirect_tqdm():
        start_time = datetime.now()
        for k in tqdm(range(K), total=K):
            if k % M == 0:
                # create MPC inputs
                vars_init, pars_val = {}, {}
                for o in origins:
                    vars_init[f'w_{o}'] = shift(vars_last[f'w_{o}'], n=M)
                    pars_val[f'd_{o}'] = get_demand(o, k)
                    pars_val[f'w0_{o}'] = o.queue[k]
                for l in links:
                    vars_init[f'rho_{l}'] = shift(vars_last[f'rho_{l}'], n=M)
                    vars_init[f'v_{l}'] = shift(vars_last[f'v_{l}'], n=M)
                    pars_val[f'rho0_{l}'] = l.density[k]
                    pars_val[f'v0_{l}'] = l.speed[k]
                for o in onramps:
                    vars_init[f'r_{o}'] = shift(vars_last[f'r_{o}'], n=1)
                    pars_val[f'r_{o}_last'] = vars_last[f'r_{o}'][0, 0]
                for l in links_vms:
                    vars_init[f'v_ctrl_{l}'] = shift(vars_last[f'v_ctrl_{l}'],
                                                     n=1)
                    pars_val[f'v_ctrl_{l}_last'] = \
                        vars_last[f'v_ctrl_{l}'][:, 0]
                for slackname, slack in MPC.slacks.items():
                    vars_init[slackname] = np.zeros(slack.shape)

                # vars_init, pars_val = dict(vars_last), {}
                # for o in origins:
                #     pars_val[f'd_{o}'] = get_demand(o, k)
                #     pars_val[f'w0_{o}'] = o.queue[k]
                # for l in links:
                #     pars_val[f'rho0_{l}'] = l.density[k]
                #     pars_val[f'v0_{l}'] = l.speed[k]
                # for o in onramps:
                #     pars_val[f'r_{o}_last'] = vars_last[f'r_{o}'][0, 0]
                # for l in links_vms:
                #     pars_val[f'v_ctrl_{l}_last'] = vars_last[f'v_ctrl_{l}'][:, 0]
                # for slackname, slack in MPC.slacks.items():
                #     vars_init[slackname] = np.zeros(slack.shape)

                # run MPC
                vars_last, info = MPC(vars_init, pars_val)
                if 'error' in info:
                    logger.error(
                        fmt(k, K, t[k], f'({name}) ' + info['error']))
                else:
                    logger.info(fmt(k, K, t[k]))

            # set onramp metering rate and vms speed control
            for onramp in onramps:
                onramp.rate[k] = vars_last[f'r_{onramp}'][0, 0]
            for l in links_vms:
                l.v_ctrl[k] = \
                    vars_last[f'v_ctrl_{l}'][:, 0].reshape((l.nb_vms, 1))

            # step the system
            # example of F args and outs
            # args: w_O1;w_O2|rho_L1[4],v_L1[4];rho_L2[2],v_L2[2]|r_O2;v_ctrl_L1[2]|d_O1;d_O2
            # outs: q_O1,w+_O1;q_O2,w+_O2|q_L1[4],rho+_L1[4],v+_L1[4];q_L2[2],rho+_L2[2],v+_L2[2]
            x, u, d = [], [], []
            for origin in origins:
                x.append(origin.queue[k])
            for link in links:
                x.append(link.density[k])
                x.append(link.speed[k])
            for onramp in onramps:
                u.append(onramp.rate[k])
            for link in links_vms:
                u.append(link.v_ctrl[k])
            for origin in origins:
                d.append(origin.demand[k])

            outs = F(*x, *u, *d)

            i = 0
            for origin in origins:
                origin.flow[k] = outs[i]
                origin.queue[k + 1] = outs[i + 1]
                i += 2
            for link in links:
                link.flow[k] = outs[i]
                link.density[k + 1] = outs[i + 1]
                link.speed[k + 1] = outs[i + 2]
                i += 3

            # save other dynamic quantities (slacks, etc.)
            sim.objective.append(info['f'])
            for slack in MPC.slacks:
                sim.slacks[slack].append(np.array(vars_last[slackname]))

            # at the end of each iteration, call the callbacks
            for cb in cbs:
                cb(k, sim, vars_last, info)  # arguments to be defined

        return datetime.now() - start_time
