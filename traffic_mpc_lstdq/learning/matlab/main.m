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

% network size
n_origins = 2;
n_ramps = 1;
n_links = 3;

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

% parameters (constant)
Np = 7;                             % prediction horizon
Nc = 3;                             % control horizon
M = 6;                              % horizon spacing factor
plugin_opts = struct('expand', false, 'print_time', false);
solver_opts = struct('print_level', 0, 'max_iter', 3e3, 'tol', 1e-5);
rate_var_penalty = 0.4;             % penalty weight for rate variability
discount = 1;                       % rl discount factor
lr = 1e-1;                          % rl learning rate
con_violation_penalty = 30;         % rl penalty for constraint violations
rl_update_freq = round(K / 2);      % when rl should update
save_freq = 5;                      % checkpoint saving

% cost terms (learnable MPC, metanet and RL)
[Vcost, Lcost, Tcost] = rlmpc.get_mpc_learnable_costs(...
    n_origins, n_links, ...
    2 * max_queue, rho_max, 1.2 * v_free, ... % normalization are constant
    'constant', 'posdef', 'posdef');
TTS = metanet.TTS(n_origins, n_links, T, L, lanes);
rate_var = metanet.rate_variability(n_ramps, Nc);
Lrl = rlmpc.get_rl_cost(n_origins, n_links, TTS, ...
    max_queue, con_violation_penalty);

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

    % create required parameters
    weight_V = ctrl.add_par('weight_V', size(Vcost.mx_in(3)));
    weight_L_rho = ctrl.add_par('weight_L_rho', size(Lcost.mx_in(4)));
    weight_L_v = ctrl.add_par('weight_L_v', size(Lcost.mx_in(5)));
    weight_T = ctrl.add_par('weight_T', size(Tcost.mx_in(2)));
    weight_slack = ctrl.add_par('weight_slack', 1, 1);
    r_last = ctrl.add_par('r_last', size(ctrl.vars.r, 1), 1);

    % compute the actual symbolic cost expression with opti vars and pars
    % initial + terminal + penalties term
    cost = Vcost(ctrl.vars.w(:, 1), ctrl.vars.rho(:, 1), ...            % affine/constant initial cost
            ctrl.vars.v(:, 1), weight_V) + ...                              
        Tcost(ctrl.vars.rho(:, end), ctrl.pars.rho_crit, weight_T) + ...% quadratic terminal cost
        weight_slack * slack(end) + ...                                 % terminal slack penalty
        rate_var_penalty * rate_var(r_last, ctrl.vars.r);               % terminal rate variability
    % stage terms
    for k = 1:M * Np
        cost = cost + ...
            weight_slack * slack(k) + ...                               % slack penalty
            Lcost(ctrl.vars.rho(:, k), ctrl.vars.v(:, k), ...           % quadratic stage cost
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

% V approximator has perturbation to enhance exploration
pert = mpc.V.add_par('perturbation', size(mpc.V.vars.r(:, 1)));
mpc.V.set_cost(mpc.V.opti.f + pert' * mpc.V.vars.r(:, 1));



%% Simulation
% initial conditions
r = 1;                                  % ramp metering rate
r_prev = 1;                             % previous rate
[w, rho, v] = util.steady_state(F, ...  % queue, density, speed
    [0; 0], [5; 5; 18], [100; 198; 90], r, D(:, 1), ...
    true_pars.a, true_pars.v_free, true_pars.rho_crit);

% initial function approx weights
rl_pars.v_free = {v_free};
rl_pars.rho_crit = {rho_crit};
rl_pars.weight_V = {ones(size(weight_V))};
rl_pars.weight_L_rho = {ones(size(weight_L_rho))};
rl_pars.weight_L_v = {ones(size(weight_L_v))};
rl_pars.weight_T = {ones(size(weight_T))};
rl_pars.weight_slack = {ones(size(weight_slack)) * 10};

% preallocate containers for miscellaneous quantities
slack = cell(1, episodes);          % of which mpc? V probably
td_error = cell(1, episodes);
exec_times = nan(1, episodes);

% preallocate containers for traffic data (don't use struct(var, cells))
origins.queue = cell(1, episodes);
origins.flow = cell(1, episodes);
origins.rate = cell(1, episodes);
links.flow = cell(1, episodes);
links.density = cell(1, episodes);
links.speed = cell(1, episodes);

% precompute Q lagrangian and its derivative w.r.t. learnable params
Qlagr = mpc.Q.opti.f + mpc.Q.opti.lam_g' * mpc.Q.opti.g;
for name = fieldnames(rl_pars)'
    dQlagr.(name{1}) = jacobian(Qlagr, mpc.Q.pars.(name{1}))';
end

% initialize mpc last solutions
last_sol = struct('w', repmat(w, 1, M * Np + 1), ...
    'r', ones(size(mpc.V.vars.r)), 'rho', repmat(rho, 1, M * Np + 1), ...
    'v', repmat(v, 1, M * Np + 1), 'slack', zeros(size(mpc.V.vars.slack)));

% simulate episodes
replaymem = rlmpc.ReplayMem(1e3);
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
    slack{ep} = nan(size(mpc.V.vars.slack, 2), ceil(K / M));
    td_error{ep} = nan(1, ceil(K / M));

    % simulate episode (do RL update at specific iterations)
    start_ep_time = tic;
    for k = 1:K
        % check if MPCs must be run
        if mod(k, M) == 1
            k_mpc = ceil(k / M); % mpc iteration

            % run Q(s_k, a_k) - excluding very first iteration
            if k_mpc > 1
                pars = struct('d', util.get_future_demand(D, k - M, K, M, Np), ...
                    'w0', w_prev, 'rho0', rho_prev, 'v0', v_prev, ...
                    'a', a, 'v_free', rl_pars.v_free{end}, ...
                    'rho_crit', rl_pars.rho_crit{end}, ...
                    'weight_V', rl_pars.weight_V{end}, ...
                    'weight_L_rho', rl_pars.weight_L_rho{end}, ...
                    'weight_L_v', rl_pars.weight_L_v{end}, ....
                    'weight_T', rl_pars.weight_T{end}, ...
                    'weight_slack', rl_pars.weight_slack{end}, ...
                    'r_last', r_prev_prev, ... % because any MPC needs prev action to the step it is solving
                    'r_first', r_prev); % because Q is solved at the previous step
                last_sol.slack = zeros(size(mpc.V.vars.slack));
                [last_sol, info_Q] = mpc.Q.solve(pars, last_sol);
            end

            % run V(s_k)
            pars = struct('d', util.get_future_demand(D, k, K, M, Np), ...
                'w0', w, 'rho0', rho, 'v0', v, ...
                'a', a, 'v_free', rl_pars.v_free{end}, ...
                'rho_crit', rl_pars.rho_crit{end}, ...
                'weight_V', rl_pars.weight_V{end}, ...
                'weight_L_rho', rl_pars.weight_L_rho{end}, ...
                'weight_L_v', rl_pars.weight_L_v{end}, ...
                'weight_T', rl_pars.weight_T{end}, ...
                'weight_slack', rl_pars.weight_slack{end}, ...
                'r_last', r, 'perturbation', (50 * 0.8^(ep - 1)) * randn);
            last_sol.slack = zeros(size(mpc.V.vars.slack));
            [last_sol, info_V] = mpc.V.solve(pars, last_sol);

            % save to memory if no error; or log error 
            if k_mpc > 1
                if info_V.success && info_Q.success
                    % compute td error
                    td_error{ep}(k_mpc) = full(Lrl(w_prev, rho_prev)) + ...
                        discount * info_V.f  - info_Q.f;
    
                    % compute gradient of Q w.r.t. params
                    for name = fieldnames(rl_pars)'
                        dQ.(name{1}) = info_Q.sol_obj.value(dQlagr.(name{1}));
                    end
    
                    % store everything in memory (as array takes much less space)
                    replaymem.add([td_error{ep}(k_mpc); ...
                        cell2mat(struct2cell(dQ))]);
                else
                    msg = '';
                    if ~info_V.success
                        msg = sprintf('V: %s. ', info_V.error);
                    end
                    if ~info_Q.success
                        msg = append(msg, sprintf('Q: %s.', info_V.error));
                    end
                    util.logging(toc(start_tot_time), ep, ...
                        toc(start_ep_time), t(k), k, K, msg);
                end
            end

            % get optimal a_k from V(s_k)
            r = last_sol.r(:, 1);
            slack{ep}(:, k_mpc) = last_sol.slack';

            % save for next transition
            r_prev_prev = r_prev;
            r_prev = r;
            w_prev = w;
            rho_prev = rho;
            v_prev = v;
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
        if mod(k, rl_update_freq) == 0
            % first row is td error, then derivatives of Q w.r.t. weigths
            exp = cell2mat(replaymem.sample(300));
            td_err = exp(1, :);

            % compute next parameters: w = w - lr * mean(td * dQ.theta) 
            i = 2;
            for name = fieldnames(rl_pars)'
                sz = length(rl_pars.(name{1}){end});
                rl_pars.(name{1}){end + 1} = rl_pars.(name{1}){end} - ...
                    lr * (exp(i:i+sz-1, :) * td_err') / length(td_err);
                i = i + sz;
            end
        end
    end
    exec_times(ep) = toc(start_ep_time);
    
    % perform some final conversions
    slack{ep} = cell2mat(slack{ep});

    % save every episode in a while
    if mod(ep - 1, save_freq) == 0
        warning('off');
        save('checkpoint.mat')
        warning('on');
    end

    % log intermediate results
    msg = sprintf('episode %i terminated: TTS=%.4f', ...
        ep, sum(TTS(origins.queue{ep}, links.density{ep})));
    util.logging(toc(start_tot_time), ep, exec_times(ep), t(end), K, K, msg);
end
exec_time_tot = toc(start_tot_time);
diary off



%% Saving and plotting
% build arrays
rl_pars = structfun(@(x) cell2mat(x), rl_pars, 'UniformOutput', false);

% clear useless variables
clear cost ctrl d d1 d2 ep F i info k last_sol log_filename msg name pars ...
    q q_o r r_first r_last r_prev replaymem rho rho_next rho_prev save_freq ...
    start_ep_time start_tot_time sz v v_next v_prev w w_next w_prev

% save
warning('off');
save(strcat(runname, '_data.mat'));
warning('on');
run visualization.m
