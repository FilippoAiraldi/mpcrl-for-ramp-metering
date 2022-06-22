% made with MATLAB 2021b
clc, clearvars, close all, diary off, warning('on')
rng(42)
runname = datestr(datetime, 'yyyymmdd_HHMMSS');
save_freq = 2;                          % checkpoint saving frequency



%% Training Environment
iterations = 2;                         % simulation iterations
episodes = 5;                          % number of episodes per iteration
[sim, mdl, mpc] = util.get_pars();

% create gym  environment with monitor
env = METANET.TrafficEnv(episodes, sim, mdl, mpc);
env = METANET.TrafficMonitor(env, iterations);

% create known (wrong) model parameters
known_pars = struct('a', env.env.model.a * 1.3, ...
                    'v_free', env.env.model.v_free * 1.3, ...
                    'rho_crit', env.env.model.rho_crit * 0.7);



%% MPC-based Q-Learning Agent
agent = RL.QLAgent(env.env, known_pars);
% TODO: ... create a monitor for the agent's quantities...
%       1. saves Qf and Qv for each MPC solving
%       2. saves transition quantities
%       3. saves update quantities
replaymem = RL.ReplayMem(mpc.mem_cap, 'sum', 'g', 'H');


%% Simulation
logger = util.Logger(env, agent, runname, false);
r = 575;        % first action
r_prev = r;     % action before that
for i = 1:iterations
    % reset initial conditions and demand
    state = env.reset(r);
    done = false;
    k_mpc = 1;

    % simulate all episodes in current iteration
    while ~done
        ep = env.env.ep;

        % compute Q(s, a)
        pars = struct('a', known_pars.a, 'r_last', r_prev, 'r0', r);
        [~, ~, infoQ] = agent.solve_Q(pars, state);

        % step dynamics to next mpc iteration
        L = 0;
        for m = 1:mpc.M
            [state_next, cost, done, info_env] = env.step(r);
            L = L + cost;
            if done; break; end
        end

        % compute V(s+)
        p = 0 * agent.rand_perturbation((i - 1) * episodes + ep);
        pars = struct('a', known_pars.a, 'r_last', r, 'perturbation', p);
        [r_next, ~, infoV] = agent.solve_V(pars, state_next);
        
        % if both are success, save transition quantities to replay memory
        % (skip a couple of first transitions)
        if k_mpc > 1
            agent.save_transition(replaymem, L, infoQ, infoV);
        end
        logger.log_mpc_status(infoQ, infoV);

        % perform RL update (do not update in the very first episode)
        if ~mod(k_mpc, mpc.update_freq) % && (i > 1 || ep > 1)
            [n, ~, Hmod] = agent.update(replaymem);
            logger.log_agent_update(-1, n, Hmod);
        end

        % episode is done, so print summary, live-plot, reset quantites
        if info_env.ep_done
            % log and plot
            logger.log_ep_recap(ep);

            % reset
            agent.Q.failures = 0;
            agent.V.failures = 0;
            env.env.reset_cumcost();
        end

        % increment/save for next timestep
        k_mpc = k_mpc + 1;
        state = state_next;
        r_prev = r;
        r = r_next;        
    end
end
