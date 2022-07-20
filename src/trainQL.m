% made with MATLAB 2021b
clc, clear all, close all, diary off, warning('on') %#ok<CLALL>
rng(42)
runname = datestr(datetime, 'yyyymmdd_HHMMSS_train');



%% Training Environment
iterations = 1;                         % simulation iterations
episodes = 75;                          % number of episodes per iteration
[sim, mdl, mpc] = util.get_pars();

% create gym environment with monitor
env = METANET.TrafficEnv(episodes, sim, mdl, mpc);
env = METANET.TrafficMonitor(env, iterations);

% create known (wrong) model parameters
known_pars = struct('a', env.env.model.a * 1.3, ...
                    'v_free', env.env.model.v_free * 1.3, ...
                    'rho_crit', env.env.model.rho_crit * 0.7);



%% MPC-based Q-Learning Agent
agent = RL.QLAgent(env.env, known_pars);
agent = RL.AgentMonitor(agent);
replaymem = RL.ReplayMem(mpc.mem_cap, 'sum', 'g', 'H');



%% Simulation
logger = util.Logger(env, agent, runname);
plotter = util.TrainLivePlot(env, agent);

for i = 1:iterations
    % reset initial conditions
    r = 575;                % first action
    r_prev = r;             % action before that
    state = env.reset(r);   % reset steady-state and demands
    done = false;           % flag to stop simulation
    k_mpc = 1;              % mpc iteration

    % simulate all episodes in current iteration
    while ~done
        ep = env.env.ep;

        % compute Q(s, a)
        pars = struct('a', known_pars.a, 'r_last', r_prev, 'r0', r);
        [~, ~, infoQ] = agent.solve_mpc('Q', pars, state);

        % step dynamics to next mpc iteration
        L = 0;
        for m = 1:mpc.M
            [state_next, cost, done, info_env] = env.step(r);
            L = L + cost;
            if done; break; end
        end

        % compute V(s+)
        pars = struct('a', known_pars.a, 'r_last', r, 'perturbation', ...
            agent.agent.rand_perturbation((i - 1) * episodes + ep));
        [r_next, ~, infoV] = agent.solve_mpc('V', pars, state_next);
        
        % if both are success, save transition quantities to replay memory
        % (skip a couple of first transitions)
        if k_mpc > 1
            agent.save_transition(replaymem, L, infoQ, infoV);
        end
        logger.log_mpc_status(infoQ, infoV);

        % perform RL update (do not update in the very first episode)
        if ~mod(k_mpc, mpc.update_freq) && (i > 1 || ep > 1)
            nb_samples = agent.update(replaymem);
            logger.log_agent_update(nb_samples);
        end

        % if episode is done, print summary, live-plot, reset quantites
        if info_env.ep_done
            % log and plot
            logger.log_ep_summary(ep);
            plotter.plot_episode_summary(i, ep);

            % reset episode quantities
            agent.agent.Q.failures = 0;
            agent.agent.V.failures = 0;
            env.env.reset_cumcost();
        end

        % increment/save for next timestep
        k_mpc = k_mpc + 1;
        state = state_next;
        r_prev = r;
        r = r_next;        
    end
end



%% Saving and plotting
warning('off');
save(strcat(runname, '_data.mat'));
warning('on');
env.plot_traffic(runname)
env.plot_cost(runname)
agent.plot_learning(runname)
