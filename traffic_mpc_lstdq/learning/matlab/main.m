clc, close all, clear all



%% logging and saving
runname = datestr(datetime,'yyyymmdd_HHMMSS');
log_filename = strcat(runname, '_log.txt');
result_filename = strcat(runname, '_data.mat');



%% Model
% simulation
episodes = 50;                  % number of episodes to repeat
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

% true (unknown) and known (wrong) model parameters
true_pars.a = 1.867;            % model parameter (adim)
true_pars.v_free = 102;         % free flow speed (km/h)
true_pars.rho_crit = 33.5;      % critical capacity (veh/km/lane)
a = 2.111;
v_free = 130;
rho_crit = 27;



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

% build mpc-based value function approximators
Np = 7;
Nc = 3;
M = 6;
plugin_opts = struct('expand', false, 'print_time', false);
solver_opts = struct('print_level', 0, 'max_iter', 3e3);
slack_penalty = 1e1;
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
    [Vcost, Lcost, Tcost] = rlmpc.get_learnable_costs( ...
        size(ctrl.vars.w, 1), size(ctrl.vars.v, 1), ...
        2 * max_queue, rho_max, 1.2 * v_free, ... % normalization are constant
        T, L, lanes, ...
        'constant', 'posdef', 'posdef');

    % create required parameters
    weight_V = ctrl.add_par('weight_V', size(Vcost.mx_in(3)));
    weight_L_rho = ctrl.add_par('weight_L_rho', size(Lcost.mx_in(5)));
    weight_L_v = ctrl.add_par('weight_L_v', size(Lcost.mx_in(6)));
    weight_T = ctrl.add_par('weight_T', size(Tcost.mx_in(2)));
    r_last = ctrl.add_par('r_last', size(ctrl.vars.r, 1), 1);

    % compute the actual symbolic cost expression with opti vars and pars
    cost = ...
        Vcost(ctrl.vars.w(:, 1), ctrl.vars.rho(:, 1), ...
            ctrl.vars.v(:, 1), weight_V) + ...                              % initial cost
        Tcost(ctrl.vars.rho(:, end), ctrl.pars.rho_crit, weight_T) + ...    % terminal cost
        slack_penalty * slack(end) + ...                                    % terminal slack penalty
        metanet.rate_variability(r_last, ctrl.vars.r);                      % terminal rate variability
    for k = 1:M * Np
        cost = cost + ...
            slack_penalty * slack(k) + ...                                  % stage slack penalty
            Lcost(ctrl.vars.w(:, k), ctrl.vars.rho(:, k), ...               
                ctrl.vars.v(:, k), ctrl.pars.rho_crit, ...
                ctrl.pars.v_free, weight_L_rho, weight_L_v);                % stage cost (includes TTS)
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
% create experience buffer
replaymem = rlmpc.ReplayMem(1e3);

% preallocate containers for data
origins.queue = cell(1, episodes);
origins.flow = cell(1, episodes);          
origins.rate = cell(1, episodes);
links.flow = cell(1, episodes);
links.density = cell(1, episodes);
links.speed = cell(1, episodes);
objective = cell(1, episodes);      % of which mpc? V probably
slack = cell(1, episodes);          % of which mpc? V probably
exec_times = cell(1, episodes);
learned_pars.a = {a};
learned_pars.v_free = {v_free};
learned_pars.rho_crit = {rho_crit};

% initial conditions
r = 1;                                  % ramp metering rate
[w, rho, v] = util.steady_state(F, ...  % queue, density, speed
    [0; 0], [5; 5; 18], [100; 198; 90], r, D(:, 1), ...
    true_pars.a, true_pars.v_free, true_pars.rho_crit);

% initialize last solutions
last_sol.w = repmat(w, 1, M * Np + 1);
last_sol.r = ones(size(mpc.V.vars.r, 1), Nc);
last_sol.rho = repmat(rho, 1, M * Np + 1);
last_sol.v = repmat(v, 1, M * Np + 1);

% simulation details
rl_update_at = [round(K / 2), K];
save_freq = 5;

% simulate episodes
% diary(log_filename)
fprintf(['# Fields: [Realtime_tot|Episode_n|Realtime_episode] ', ...
    '- [Sim_time|Sim_iter|Sim_perc] - Message\n'])
start_tot_time = tic;
for i = 1:episodes
    % preallocate episode result containers
    origins.queue{i} = nan(size(mpc.V.vars.w, 1), K + 1);        
    origins.flow{i} = nan(size(mpc.V.vars.w, 1), K);          
    origins.rate{i} = nan(size(mpc.V.vars.r, 1), K);
    links.flow{i} = nan(size(mpc.V.vars.v, 1), K);
    links.density{i} = nan(size(mpc.V.vars.rho, 1), K + 1);
    links.speed{i} = nan(size(mpc.V.vars.v, 1), K + 1);
    objective{i} = {}; 
    slack{i} = {};

    % simulate episode (do RL update at specific iterations)
    start_ep_time = tic;
    for k = 1:K
        % compute MPC control action 
        if mod(k, M) == 1
            % assemble inputs
            d = util.get_future_demand(D, k, K, M, Np);

            % run MPC
            % ...

            % print if error
            % ...

            % assign results
            r = 1;
            objective{i}{end + 1} = 1;
            slack{i}{end + 1} = ones(1, M * Np + 1)';

            % store transition in replay memory (from M states ago) only if
            % no errors occurred
            % ...
        end

        % step state (according to the true dynamics)
        [q_o, w_next, q, rho_next, v_next] = F(w, rho, v, r, D(:, k), ...
            true_pars.a, true_pars.v_free, true_pars.rho_crit);

        % assign current state
        origins.queue{i}(:, k) = full(w);
        origins.flow{i}(:, k) = full(q_o);
        origins.rate{i}(:, k) = full(r);
        links.flow{i}(:, k) = full(q);
        links.density{i}(:, k) = full(rho);
        links.speed{i}(:, k) = full(v);

        % set next state as current
        w = full(w_next);
        rho = full(rho_next);
        v = full(v_next);
        
        % perform RL updates
        if ismember(k, rl_update_at)
            % ...
        end
    end
    exec_times{i} = toc(start_ep_time);
    
    % perform some final conversions
    objective{i} = cell2mat(objective{i});
    slack{i} = cell2mat(slack{i});

    % save every episode in a while
    if mod(i - 1, 2) == 0
        warning('off');
        save('checkpoint.mat')
        warning('on');
    end

    % log intermediate results -> sum of objective at least
    msg = sprintf('episode %i terminated: cost=%.3f', i, sum(objective{i}));
    util.logging(toc(start_tot_time), i, exec_times{i}, t(end), K, K, msg);
end
diary off



%% Plotting
