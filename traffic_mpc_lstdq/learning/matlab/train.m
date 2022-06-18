% made with MATLAB 2021b
clc, clearvars, close all, diary off, warning('on')
rng(42)
runname = datestr(datetime, 'yyyymmdd_HHMMSS');
save_freq = 2;                          % checkpoint saving frequency



%% Training Environment
iterations = 3;                         % simulation iterations
episodes = 7;                          % number of episodes per iteration
[sim, mdl, mpc] = util.get_pars();

% create gym  environment with monitor
env = METANET.TrafficEnv(episodes, sim, mdl, mpc);
env = METANET.TrafficMonitor(env, iterations);

% create known (wrong) model parameters
known_mdl_pars = struct('a', env.env.model.a * 1.3, ...
                        'v_free', env.env.model.v_free * 1.3, ...
                        'rho_crit', env.env.model.rho_crit * 0.7);



%% MPC-based Q-Learning Agent
agent = RL.QAgent(env.env, known_mdl_pars);
return


%% Simulation
for i = 1:iterations
    % reset initial conditions and demand
    env.reset(575);
    done = false;

    % simulate all episodes
    while ~done
        % choose control actiobn
        r = max(575, 900 + 100 * randn);

        % step the system
        [state, cost, done, info] = env.step(r);
        if ~done
            assert(i == env.iter);
        end

        % check if the single episode is done
        if info.ep_done
            fprintf('ep=%i, k=%i, k_tot=%i\n', env.env.ep, env.env.k, env.env.k_tot);
        end
    end
end
