% run with MATLAB 2021b
clc, close all, clear all
runname = datestr(datetime,'yyyymmdd_HHMMSS');



%% Model
% simulation
episodes = 02;                  % number of episodes to repeat
Tfin = 2;                       % simulation time per episode (h)
T = 10 / 3600;                  % simulation step size (h)
K = Tfin / T;                   % simulation steps per episode (integer)
t = (0:(K - 1)) * T;            % time vector (h) (only first episode)

% segments
L = 1;                          % length of links (km)
lanes = 2;                      % lanes per link (adim)

% on-ramp O2
C2 = 2000;                      % on-ramp capacity (veh/h/lane)
max_queue = 100;

% model parameters
tau = 18 / 3600;                % model parameter (s)
kappa = 40;                     % model parameter (veh/km/lane)
eta = 60;                       % model parameter (km^2/lane)
rho_max = 180;                  % maximum capacity (veh/km/lane)
delta = 0.0122;                 % merging phenomenum parameter

% known (wrong) and true (unknown) model parameters
a = 2.111;                      % model parameter (adim)
v_free = 130;                   % free flow speed (km/h)
rho_crit = 27;                  % critical capacity (veh/km/lane)
true_pars = struct('a', 1.867, 'v_free', 102, 'rho_crit', 33.5);



%% Disturbances
d1 = util.create_profile(t, [0, .35, 1, 1.35], [1000, 3150, 3150, 1000]);
d2 = util.create_profile(t, [.15, .35, .6, .8], [500, 1800, 1800, 500]);
D = [d1; d2];
% plot(t, d1, t, d2),
% legend('O1', 'O2'), xlabel('time (h)'), ylabel('demand (h)')
% ylim([0, 4000])



%% MPC-based RL
% create a symbolic casadi function for the dynamics
F = metanet.get_dynamics(T, L, lanes, C2, rho_max, tau, delta, eta, kappa);

% parameters
Np = 7;
Nc = 3;
M = 6;
plugin_opts = struct('expand', false, 'print_time', false);
solver_opts = struct('print_level', 0, 'max_iter', 3e3);
rate_var_penalty = 0.4;
discount = 1;   % rl discount factor
lr = 1e-3;      % rl learning rate

% build mpc-based value function approximators
mpc = struct;
for name = ["Q", "V"]
    % instantiate an MPC
    ctrl = rlmpc.NMPC(Np, Nc, M);
    ctrl.init_opti(F); 

    % set solver
    ctrl.set_ipopt_opts(plugin_opts, solver_opts);

    % set soft constraint on ramp queue
    slack = ctrl.add_var('slack', 1, M * Np + 1);
    ctrl.opti.subject_to(slack >= 0);
    ctrl.opti.subject_to(ctrl.vars.w(2, :) - slack <= max_queue)

    % get the costs components
    n_origins = size(ctrl.vars.w, 1);
    n_links = size(ctrl.vars.v, 1);
    [Vcost, Lcost, Tcost] = rlmpc.get_learnable_costs(...
        n_origins, n_links, ...
        2 * max_queue, rho_max, 1.2 * v_free, ... % normalization are constant
        'constant', 'posdef', 'posdef');
    TTS = metanet.TTS(n_origins, n_links, T, L, lanes);
    rate_var = metanet.rate_variability(size(ctrl.vars.r, 1), Nc);

    % create required parameters
    weight_V = ctrl.add_par('weight_V', size(Vcost.mx_in(3)));
    weight_L_rho = ctrl.add_par('weight_L_rho', size(Lcost.mx_in(4)));
    weight_L_v = ctrl.add_par('weight_L_v', size(Lcost.mx_in(5)));
    weight_T = ctrl.add_par('weight_T', size(Tcost.mx_in(2)));
    weight_slack = ctrl.add_par('weight_slack', 1, 1);
    r_last = ctrl.add_par('r_last', size(ctrl.vars.r, 1), 1);

    % compute the actual symbolic cost expression with opti vars and pars
    cost = Vcost(ctrl.vars.w(:, 1), ctrl.vars.rho(:, 1), ...            % initial cost
            ctrl.vars.v(:, 1), weight_V) + ...                              
        Tcost(ctrl.vars.rho(:, end), ctrl.pars.rho_crit, weight_T) + ...% terminal cost
        weight_slack * slack(end) + ...                                 % terminal slack penalty
        rate_var_penalty * rate_var(r_last, ctrl.vars.r);               % terminal rate variability
    for k = 1:M * Np
        cost = cost + ...
            weight_slack * slack(k) + ...                               % stage slack penalty
            Lcost(ctrl.vars.rho(:, k), ctrl.vars.v(:, k), ...           % stage cost
                ctrl.pars.rho_crit, ctrl.pars.v_free, ...
                weight_L_rho, weight_L_v) + ...                         
            TTS(ctrl.vars.w(:, k), ctrl.vars.rho(:, k));                % TTS
    end

    % assign cost to opti
    ctrl.set_cost(cost);

    % save to struct
    mpc.(name) = ctrl;
end

% Q approximator has additional constraint on first action
r_first = mpc.Q.add_par('r_first', 1, 1);
mpc.Q.opti.subject_to(mpc.Q.vars.r(:, 1) == r_first);


%% Simulation
% initial conditions
r = 1;                                  % ramp metering rate
[w, rho, v] = util.steady_state(F, ...  % queue, density, speed
    [0; 0], [5; 5; 18], [100; 198; 90], r, D(:, 1), ...
    true_pars.a, true_pars.v_free, true_pars.rho_crit);

% initial function approx weights
weight_V = ones(size(Vcost.mx_in(3)));
weight_L_rho = ones(size(Lcost.mx_in(4))) * 2;
weight_L_v = ones(size(Lcost.mx_in(5))) * 2;
weight_T = ones(size(Tcost.mx_in(2))) * 5;
weight_slack = 10;

% preallocate containers for data (don;t use struct(var, cells))
origins.queue = cell(1, episodes);
origins.flow = cell(1, episodes);
origins.rate = cell(1, episodes);
links.flow = cell(1, episodes);
links.density = cell(1, episodes);
links.speed = cell(1, episodes);
objective = cell(1, episodes);      % of which mpc? V probably
slack = cell(1, episodes);          % of which mpc? V probably
exec_times = cell(1, episodes);
rl_pars.v_free = {v_free};
rl_pars.rho_crit = {rho_crit};
rl_pars.weight_V = {weight_V};
rl_pars.weight_L_rho = {weight_L_rho};
rl_pars.weight_L_v = {weight_L_v};
rl_pars.weight_T = {weight_T};
rl_pars.weight_slack = {weight_slack};

% initialize last solutions
last_sol = struct('w', repmat(w, 1, M * Np + 1), ...
    'r', ones(size(mpc.V.vars.r)), 'rho', repmat(rho, 1, M * Np + 1), ...
    'v', repmat(v, 1, M * Np + 1), 'slack', zeros(size(mpc.V.vars.slack)));

% simulation details
rl_update_at = [round(K / 2), K];
save_freq = 5;
replaymem = rlmpc.ReplayMem(1e3);

% simulate episodes
% diary(strcat(runname, '_log.txt'))
fprintf(['# Fields: [Realtime_tot|Episode_n|Realtime_episode] ', ...
    '- [Sim_time|Sim_iter|Sim_perc] - Message\n'])
start_tot_time = tic;
for ep = 1:episodes
    % preallocate episode result containers
    origins.queue{ep} = nan(size(mpc.V.vars.w, 1), K);        
    origins.flow{ep} = nan(size(mpc.V.vars.w, 1), K);          
    origins.rate{ep} = nan(size(mpc.V.vars.r, 1), K);
    links.flow{ep} = nan(size(mpc.V.vars.v, 1), K);
    links.density{ep} = nan(size(mpc.V.vars.rho, 1), K);
    links.speed{ep} = nan(size(mpc.V.vars.v, 1), K);
    objective{ep} = {}; 
    slack{ep} = {};

    % simulate episode (do RL update at specific iterations)
    start_ep_time = tic;
    for k = 1:K
        % compute MPC control action 
        if mod(k, M) == 1
            % assemble inputs
            d = util.get_future_demand(D, k, K, M, Np);

            % run MPC
            pars = struct('d', d, 'w0', w, 'rho0', rho, 'v0', v, ...
                'a', a, 'v_free', v_free, 'rho_crit', rho_crit, ...
                'weight_V', weight_V, 'weight_L_rho', weight_L_rho, ...
                'weight_L_v', weight_L_v, 'weight_T', weight_T, ...
                'weight_slack', weight_slack, 'r_last', r);
            % last_sol.slack = zeros(size(mpc.V.vars.slack));
            [last_sol, info] = mpc.V.solve(pars, last_sol);

            % check if error during solver
            if ~info.success
                util.logging(toc(start_tot_time), ep, ...
                    toc(start_ep_time), t(k), k, K, info.error);
            else
                % store transition in replay memory (from M states ago) 
                % only if no errors occurred in this and previous iteration

            end

            % assign results
            r = last_sol.r(:, 1);
            objective{ep}{end + 1} = info.f;
            slack{ep}{end + 1} = last_sol.slack';
        end

        % step state (according to the true dynamics)
        [q_o, w_next, q, rho_next, v_next] = F(w, rho, v, r, D(:, k), ...
            true_pars.a, true_pars.v_free, true_pars.rho_crit);

        % save current state
        origins.queue{ep}(:, k) = full(w);
        origins.flow{ep}(:, k) = full(q_o);
        origins.rate{ep}(:, k) = full(r);
        links.flow{ep}(:, k) = full(q);
        links.density{ep}(:, k) = full(rho);
        links.speed{ep}(:, k) = full(v);

        % set next state as current
        w = full(w_next);
        rho = full(rho_next);
        v = full(v_next);
        
        % perform RL updates
        if ismember(k, rl_update_at)
            % ...
        end
    end
    exec_times{ep} = toc(start_ep_time);
    
    % perform some final conversions
    objective{ep} = cell2mat(objective{ep});
    slack{ep} = cell2mat(slack{ep});

    % save every episode in a while
    if mod(ep - 1, save_freq) == 0
        warning('off');
        save('checkpoint.mat')
        warning('on');
    end

    % log intermediate results -> sum of objective at least
    msg = sprintf('episode %i terminated: cost=%.4f (TTS=%.4f)', ep, ...
        sum(objective{ep}), sum(TTS(origins.queue{ep}, links.density{ep})));
    util.logging(toc(start_tot_time), ep, exec_times{ep}, t(end), K, K, msg);
end
exec_time_tot = toc(start_tot_time);
diary off



%% Saving and plotting
% build arrays
rl_pars = structfun(@(x) cell2mat(x), rl_pars, 'UniformOutput', false);

% clear useless variables
clear cost ctrl d d1 d2 ep F info k last_sol log_filename msg name pars ...
    q q_o r r_first r_last replaymem rho rho_next ...
    save_freq start_ep_time start_tot_time v v_next w w_next

% save
warning('off');
save(strcat(runname, '_data.mat'));
warning('on');
run visualization.m
