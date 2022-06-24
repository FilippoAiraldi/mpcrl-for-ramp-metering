% made with MATLAB 2021b
clc, clear all, close all, diary off, warning('on') %#ok<CLALL>
rng(69)
runname = [datestr(datetime, 'yyyymmdd_HHMMSS'), '_valid'];
eval_agents = {           % paths to the agents to evaluate
    'QL', 'test_for_validation.mat'; ...
    'DPG', 'test_for_validation.mat'; ...
};  
Na = size(eval_agents, 1) + 1; % add baseline agent



%% Evalution Environments
iterations = 2;                         % simulation iterations
episodes = 3;                           % number of episodes per iteration
[sim, mdl, mpc] = util.get_pars();
mpc.multistart = 1; %2 * 4;                 % make sure we are using multistart

% create gym environments with monitors - one for each agent
envs = arrayfun(@(i) METANET.TrafficMonitor( ...
        METANET.TrafficEnv(episodes, sim, mdl, mpc), iterations), 1:Na);

% create known (wrong, not learned) model parameter a
a = mdl.a * 1.3; % a is fixed to the wrong value



%% Agents
% load each agent's class and weights, then instantiate it
agents = RL.AgentMonitor.empty(0, Na);
for i = 1:Na - 1
    % load
    agentname = eval_agents{1, 1};
    warning('off');
    agentdata = load(eval_agents{i, 2}, 'agent').agent.agent;
    warning('on');

    % get class and weights
    cls = str2func(class(agentdata));
    w = agentdata.weights.value;

    % instantiate and set weights
    mdl_pars = struct('a', a, 'v_free', w.v_free, 'rho_crit', w.rho_crit);
    agents(i) = RL.AgentMonitor(cls(envs(i).env, mdl_pars, agentname));
    agents(i).agent.set_weight_values(w);
end

% instantiate baseline, perfect information agent
% .... TODO .....
agents(end + 1) = RL.AgentMonitor( ...
    cls(envs(end).env, struct('a', mdl.a, ... % true model parameters
                              'v_free', mdl.v_free, ...
                              'rho_crit', mdl.rho_crit), 'PI'));



%% Simulation
loggers = arrayfun(@(i) util.Logger( ...
        envs(i), agents(i), [runname, '_', agents(i).agent.name], false), 1:Na); % TODO: restore diary logging

pars = struct('a', a, 'r_last', [], 'perturbation', 0);
for i = 1:iterations
    % reset initial conditions
    r = 700 * ones(1, Na);          % first action
    done = false(1, Na);            % flag to stop simulation
    k_mpc = 1;                      % mpc iteration

    % reset only first env and sync it with the others
    envs(1).reset(r(1));            % reset steady-state and demands
    arrayfun(@(i) envs(i).env.sync(envs(1).env), 2:Na);

    % simulate all episodes in current iteration
    while ~any(done)
        ep = envs(1).env.ep;

        for j = 1:Na
            % compute V(s+)
            pars.r_last = r(j);
            [r(j), ~, infoV] = agents(j).solve_mpc('V', pars);
            loggers(j).log_mpc_status([], infoV);
            
            % step dynamics to next mpc iteration
            for m = 1:mpc.M
                [~, ~, done(j), info_env] = envs(j).step(r(j));
                if done(j); break; end
            end
    
            % if episode is done, print summary, live-plot, reset quantites
            if info_env.ep_done
                loggers(j).log_ep_summary(ep);
                agents(j).agent.Q.failures = 0;
                agents(j).agent.V.failures = 0;
                envs(j).env.reset_cumcost();
            end
        end

        % increment next timestep
        k_mpc = k_mpc + 1;   
    end
end



%% Saving and plotting
warning('off');
save(strcat(runname, '_data.mat'));
warning('on');
envs.plot_traffic(runname);
envs.plot_cost(runname);
