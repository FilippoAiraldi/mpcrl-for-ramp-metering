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
agent = RL.QLAgent(env.env, known_mdl_pars);



%% Simulation
r = 575; % first action
for i = 1:iterations
    % reset initial conditions and demand
    state = env.reset(r);
    done = false;
    k_mpc = 1;

    % simulate all episodes in current iteration
    while ~done % 
        % compute Q(s, a)
        % .....

        % step dynamics to next mpc iteration
        for m = 1:mpc.M
            [state_next, cost, done, info] = env.step(r);
            if done 
                break
            end
        end

        % compute V(s+)
        % .....
        r_next = max(575, 900 + 100 * randn);
        

        % increment/save for next timestep
        k_mpc = k_mpc + 1;
        state = state_next;
        r = r_next;
    end
end