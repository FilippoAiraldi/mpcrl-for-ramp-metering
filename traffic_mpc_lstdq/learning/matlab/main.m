% made with MATLAB 2021b
clc, clearvars, close all, diary off, warning('on')
rng(42)
runname = datestr(datetime, 'yyyymmdd_HHMMSS');
load_checkpoint = false;



% TO TRY
% bring modifications from 4th meeting

% THINGS TO TRY AFTER SUCCESS
% remove max(0, x) on inputs
% remove eps -> 0

% SCENARIOS
% 20220427_210530 / params 1.3,1.3,0.7; opts 3e3,5e-7: not bad but in one iteration has 6e4 of costs 
%                 / params 1.3,1.3,0.6; opts 3e3,1e-7:



%% Model
% simulation
episodes = 50;                  % number of episodes to repeat
Tfin = 2;                       % simulation time per episode (h)
T = 10 / 3600;                  % simulation step size (h)
K = Tfin / T;                   % simulation steps per episode (integer)
t = (0:(K - 1)) * T;            % time vector (h) (only first episode)

% model parameters
approx = struct;                % structure containing approximations
approx.origin_as_ramp = true;   % origin is regarded as a ramp if true; otherwise, as a pure flow with no queue
approx.control_origin = false;   % toggle this to control the origin ramp
approx.simple_rho_down = false;  % removes the max/min from the density downstream computations
approx.flow_as_control_action = true; % if true, the control action is the ramp flows; otherwise, metering rate is used
assert(approx.origin_as_ramp || ~approx.control_origin)
if approx.origin_as_ramp && approx.flow_as_control_action ...
                                                && ~approx.control_origin
    warning(['with this combo, the origin has a min function. Would ' ...
        'suggest to model the origin as a ramp and control it'])
end

% network size
n_origins = 1 + approx.origin_as_ramp;
n_links = 3;
n_ramps = 1 + approx.control_origin;                   

% segments
L = 1;                          % length of links (km)
lanes = 2;                      % lanes per link (adim)

% origins O1 and O2 
C = [3500, 2000];               % on-ramp capacity (veh/h/lane)
max_queue = [150, 50];          % maximum queue (veh) - for constraint
if ~approx.control_origin
    max_queue(1) = inf;
end
if ~approx.origin_as_ramp
    C = C(2);
    max_queue = max_queue(2);
end

% model parameters
tau = 18 / 3600;                % model parameter (s)
kappa = 40;                     % model parameter (veh/km/lane)
eta = 60;                       % model parameter (km^2/lane)
rho_max = 180;                  % maximum capacity (veh/km/lane)
delta = 0.0122;                 % merging phenomenum parameter

% known (wrong) and true (unknown) model parameters
% a = 2.111;                      % model parameter (adim)
% v_free = 130;                   % free flow speed (km/h)
% rho_crit = 27;                  % critical capacity (veh/km/lane)
true_pars = struct('a', 1.867, 'v_free', 102, 'rho_crit', 33.5);
a = true_pars.a * 1.3;
v_free = true_pars.v_free * 1.3;
rho_crit = true_pars.rho_crit * 0.7;



%% Disturbances
d1 = util.create_profile(t, [0, .35, 1, 1.35], [1000, 3000, 3000, 1000]);
d2 = util.create_profile(t, [.15, .35, .6, .8], [500, 1500, 1500, 500]);
d_cong = util.create_profile(t, [0.5, .7, 1, 1.2], [20, 60, 60, 20]);

% add noise
D = repmat([d1; d2; d_cong], 1, episodes + 1); % +1 to avoid out-of-bound access
[filter_num, filter_den] = butter(3, 0.2);
D = filtfilt(...
    filter_num, filter_den, (D + randn(size(D)) .* [75; 75; 1.5])')';

% plot((0:length(D) - 1) * T, (D .* [1; 1; 50])'),
% legend('O1', 'O2', 'cong_{\times50}'), xlabel('time (h)'), ylabel('demand (h)')
% ylim([0, 4000])



%% MPC-based RL
% parameters (constant)
approx.Veq = false;                 % whether to use an approximation of Veq
max_in_and_out = [false, true];    % whether to apply max to inputs and outputs of dynamics
%
Np = 4;                             % prediction horizon - \approx 3*L/(M*T*v_avg)
Nc = 3;                             % control horizon
M  = 6;                             % horizon spacing factor
eps = 0;                            % nonnegative constraint precision
opts.ipopt = struct('expand', 1, 'print_time', 0, 'ipopt', ...
                struct('print_level', 0, 'max_iter', 3e3, 'tol', 1e-8, ...
                       'barrier_tol_factor', 10));
opts.sqpmethod = struct('expand', 1, 'qpsol', 'qrqp', 'verbose', 0, ...
                        'print_header', 0, 'print_iteration', 0, ...
                        'print_status', 0, 'print_time', 0, ...
                        'qpsol_options', struct('print_iter', 0, ...
                                                'print_header', 0, ...
                                                'error_on_fail', 0));
opts.fmincon = optimoptions('fmincon', 'Algorithm', 'sqp', ...
                            'Display', 'none', ...
                            'OptimalityTolerance', 1e-7, ...
                            'StepTolerance', 1e-7, ...
                            'ScaleProblem', true, ...
                            'SpecifyObjectiveGradient', true, ...
                            'SpecifyConstraintGradient', true);
perturb_mag = 1;                    % magnitude of exploratory perturbation
if ~approx.flow_as_control_action
    rate_var_penalty = 0.4;         % penalty weight for rate variability
else
    rate_var_penalty = 0.04;
end
methods = {'ipopt', 'sqpmethod', 'fmincon'};
method = methods{1};                % solver method for MPC
multistart = 1; %4 * 3;                 % multistarting NMPC solver
soft_domain_constraints = false;     % whether to use soft constraints on positivity of states (either this, or max on output)
if ~soft_domain_constraints && ~max_in_and_out(2)
    warning('Dynamics can be negative and hard constraints unfeasible')
end
%
discount = 1;                       % rl discount factor
lr = 1e-7;                          % rl learning rate
con_violation_penalty = 10;         % penalty for constraint violations
rl_update_freq = round(K / 5);      % when rl should update
rl_mem_cap = 2 * K;                 % RL experience replay capacity
rl_mem_sample = rl_update_freq * 4; % RL experience replay sampling size
rl_mem_last = 0.25;                 % percentage of last experiences to include in sample
save_freq = 2;                      % checkpoint saving frequency

% create a symbolic casadi function for the dynamics (both true and nominal)
n_dist = size(D, 1);
args = {n_links, n_origins, n_ramps, n_dist, T, L, lanes, C, rho_max, ...
    tau, delta, eta, kappa, max_in_and_out, eps, ...
    approx.origin_as_ramp, approx.control_origin, ...
    approx.simple_rho_down, approx.flow_as_control_action};
if approx.Veq
    [Veq_approx, pars_Veq_approx] = ...
                metanet.get_Veq_approx(v_free, a, rho_crit, rho_max, eps);
    args{end + 1} = Veq_approx;
end
dynamics = metanet.get_dynamics(args{:});

% cost terms (learnable MPC, metanet and RL)
[TTS, Rate_var] = metanet.get_mpc_costs(n_links, n_origins, n_ramps, ...
                                                        Nc, T, L, lanes);
[Vcost, Lcost, Tcost] = rlmpc.get_mpc_costs(n_links, n_origins, ...
                                                'affine', 'diag', 'diag');    
Lrl = rlmpc.get_rl_cost(n_links, n_origins, TTS, ...
                                        max_queue, con_violation_penalty);
normalization.w = max(max_queue(isfinite(max_queue)));   % normalization constants
normalization.rho = rho_max;
normalization.v = v_free; 

% build mpc-based value function approximators
mpc = struct;
for name = ["Q", "V"]
    % instantiate an MPC
    ctrl = rlmpc.NMPC(name, Np, Nc, M, dynamics.nominal, max_queue, ...
                           soft_domain_constraints, eps, ...
                           approx.flow_as_control_action, rho_max, C, T);

    % grab the names of the slack variables
    slacknames = fieldnames(ctrl.vars)';
    slacknames = string(slacknames(startsWith(slacknames, 'slack')));

    % create required parameters
    ctrl.add_par('v_free_tracking', [1, 1]); 
    ctrl.add_par('weight_V', size(Vcost.mx_in( ...
                find(startsWith(string(Vcost.name_in), 'weight')) - 1)));
    ctrl.add_par('weight_L', size(Lcost.mx_in( ...
                find(startsWith(string(Lcost.name_in), 'weight')) - 1)));
    ctrl.add_par('weight_T', size(Tcost.mx_in( ...
                find(startsWith(string(Tcost.name_in), 'weight')) - 1)));
    ctrl.add_par('weight_rate_var', [1, 1]);
    if isfield(ctrl.vars, 'slack_w_max')
        ctrl.add_par('weight_slack_w_max', ...
                                    [numel(ctrl.vars.slack_w_max), 1]);
    end
    ctrl.add_par('r_last', [size(ctrl.vars.r, 1), 1]);

    % initial, stage and terminal learnable costs
    cost = Vcost(ctrl.vars.w(:, 1), ...
                 ctrl.vars.rho(:, 1), ...
                 ctrl.vars.v(:, 1), ...
                 ctrl.pars.weight_V, ...
                 normalization.w, ...
                 normalization.rho, ...
                 normalization.v);
    for k = 2:M * Np
        cost = cost + Lcost(ctrl.vars.rho(:, k), ...
                            ctrl.vars.v(:, k), ...
                            ctrl.pars.rho_crit, ...
                            ctrl.pars.v_free_tracking, ... % ctrl.pars.v_free
                            ctrl.pars.weight_L, ...
                            normalization.rho, ...
                            normalization.v);
    end
    % terminal cost
    cost = cost + Tcost(ctrl.vars.rho(:, end), ...
                        ctrl.vars.v(:, end), ...
                        ctrl.pars.rho_crit, ...
                        ctrl.pars.v_free_tracking, ...
                        ctrl.pars.weight_T, ...
                        normalization.rho, ...
                        normalization.v);
    % max queue slack cost and domain slack cost
    for n = slacknames
        if endsWith(n, 'w_max')
            cost = cost ... % could use also trace
                + ctrl.pars.weight_slack_w_max' * ctrl.vars.slack_w_max(:);
        else
            cost = cost + sum(con_violation_penalty^2 * ctrl.vars.(n)(:));
        end
    end
    % traffic-related cost
    cost = cost ...
        + sum(TTS(ctrl.vars.w, ctrl.vars.rho)) ...  % TTS
        + ctrl.pars.weight_rate_var * ...           % terminal rate variability
                                Rate_var(ctrl.pars.r_last, ctrl.vars.r);  

    % assign cost to opti
    ctrl.minimize(cost);

    % save to struct
    mpc.(name) = ctrl;
end
clear ctrl

% Q approximator has additional constraint on first action
mpc.Q.add_par('r0', [size(mpc.Q.vars.r, 1), 1]);
mpc.Q.add_con('r0_blocked', mpc.Q.vars.r(:, 1) - mpc.Q.pars.r0, 0, 0);

% V approximator has perturbation to enhance exploration
mpc.V.add_par('perturbation', size(mpc.V.vars.r(:, 1)));
mpc.V.minimize(mpc.V.f + mpc.V.pars.perturbation' * mpc.V.vars.r(:, 1));


%% Simulation
% initial conditions
if ~approx.flow_as_control_action
    r = ones(n_ramps, 1);                       % ramp metering rate
    r_prev = r;                                 % previous rate
else
    r = ones(n_ramps, 1) * max(D(:, 1)) * 1.05; % flow rate
    r_prev = r;                                 % previous rate    
end
[w, rho, v] = util.steady_state(dynamics.real.f, ...
    zeros(n_origins, 1), 10 * ones(n_links, 1), 100 * ones(n_links, 1), ...
    r, D(:, 1), true_pars.rho_crit, true_pars.a, true_pars.v_free);



% initial learnable Q/V function approx. weights and their bounds
args = cell(0, 3);
args(end + 1, :) = {'rho_crit', {rho_crit}, [10, rho_max * 0.9]};
if ~approx.Veq
    args(end + 1, :) = {'v_free', {v_free}, [30, 300]};
else
    args(end + 1, :) = {'pars_Veq_approx', ...
        {pars_Veq_approx}, [-1e2, 0; 1e-2, 2 * v_free; 1e-2, 2 * v_free]};
end
args(end + 1, :) = {'v_free_tracking', {v_free}, [30, 300]};
args(end + 1, :) = {'weight_V', ...
                        {ones(size(mpc.V.pars.weight_V))}, [-inf, inf]};
args(end + 1, :) = {'weight_L', ...
                        {ones(size(mpc.V.pars.weight_L))}, [0, inf]};
args(end + 1, :) = {'weight_T', ...
                        {ones(size(mpc.V.pars.weight_T))}, [0, inf]};
args(end + 1, :) = {'weight_rate_var', {rate_var_penalty}, [1e-3, 1e3]};
if isfield(mpc.V.vars, 'slack_w_max')
    args(end+1,:) = {'weight_slack_w_max', ...
        {ones(size(mpc.V.pars.weight_slack_w_max)) * ...
                                        con_violation_penalty}, [0, inf]};
end
rl = struct;
rl.pars = cell2struct(args(:, 2), args(:, 1));
rl.bounds = cell2struct(args(:, 3), args(:, 1));

% compute symbolic derivatives
deriv = struct;
for n = string(fieldnames(mpc)')
    % assemble all RL parameters in a single vector for V and Q
    deriv.(n).rl_pars = mpc.(n).concat_pars(fieldnames(rl.pars));

    % compute derivative of Lagrangian
    Lagr = mpc.(n).lagrangian;
    deriv.(n).dL = simplify(jacobian(Lagr, deriv.(n).rl_pars)');
    deriv.(n).d2L = simplify(hessian(Lagr, deriv.(n).rl_pars));
end

% preallocate containers for traffic data (don't use struct(var, cells))
origins.queue = cell(1, episodes);
origins.flow = cell(1, episodes);
origins.rate = cell(1, episodes);
origins.demand = cell(1, episodes);
links.flow = cell(1, episodes);
links.density = cell(1, episodes);
links.speed = cell(1, episodes);

% preallocate containers for miscellaneous quantities
slacks = struct;
for n = slacknames
    slacks.(n) = cell(1, episodes);
end
td_error = cell(1, episodes);
td_error_perc = cell(1, episodes);  % td error as a percentage of the Q function
exec_times = nan(1, episodes);

% initialize mpc last solutions to steady-state
last_sol = struct( ...
    'w', repmat(w, 1, M * Np + 1), ...
    'rho', repmat(rho, 1, M * Np + 1), ...
    'v', repmat(v, 1, M * Np + 1), ...
    'r', repmat(r, 1, Nc));
for n = slacknames
    last_sol.(n) = ones(size(mpc.V.vars.(n))) * eps^2;
end

% initialize mpc solvers
switch method
    case 'ipopt'
        args = opts.ipopt;
    case 'sqpmethod'
        args = opts.sqpmethod;
    case 'fmincon'
        args = opts.fmincon;
    otherwise
        error('invalid method')
end
mpc.Q.init_solver(args);
mpc.V.init_solver(args);

% create replay memory
replaymem = rlmpc.ReplayMem(rl_mem_cap, 'sum', 'A', 'b', 'dQ');

% load checkpoint
if load_checkpoint
    load checkpoint.mat
    start_ep = ep + 1;
else
    start_ep = 1;
end

% start logging
diary(strcat(runname, '_log.txt'))
fprintf(['# Fields: [Realtime_tot|Episode_n|Realtime_episode] ', ...
    '- [Sim_time|Sim_iter|Sim_perc] - Message\n'])

% simulate episodes
if load_checkpoint
    fprintf('Loaded checkpoint\n')
else
    start_tot_time = tic;
end
for ep = start_ep:episodes
    % preallocate episode result containers
    origins.queue{ep} = nan(size(mpc.V.vars.w, 1), K);
    origins.flow{ep} = nan(size(mpc.V.vars.w, 1), K);
    origins.rate{ep} = nan(size(mpc.V.vars.r, 1), K);
    origins.demand{ep} = nan(size(mpc.V.pars.d, 1), K);
    links.flow{ep} = nan(size(mpc.V.vars.v, 1), K);
    links.density{ep} = nan(size(mpc.V.vars.rho, 1), K);
    links.speed{ep} = nan(size(mpc.V.vars.v, 1), K);
    for n = slacknames
        slacks.(n){ep} = nan(numel(mpc.V.vars.(n)), ceil(K / M));
    end
    td_error{ep} = nan(1, ceil(K / M));
    td_error_perc{ep} = nan(1, ceil(K / M));

    % simulate episode
    start_ep_time = tic;
    nb_fail = 0;
    for k = 1:K
        % check if MPCs must be run
        if mod(k, M) == 1
            k_mpc = ceil(k / M); % mpc iteration

            % run Q(s(k-1), a(k-1)) (condition excludes very first iteration)
            if ep > 1 || k_mpc > 1
                pars = struct( ...
                    'd', D(:, K*(ep-1) + k-M:K*(ep-1) + k-M + M*Np-1), ...
                    'w0', w_prev, 'rho0', rho_prev, 'v0', v_prev, ...
                    'r_last', r_prev_prev, 'r0', r_prev); % a(k-2) and a(k-1)
                if ~approx.Veq
                    pars.a = a;
                end
                for n = fieldnames(rl.pars)'
                    pars.(n{1}) = rl.pars.(n{1}){end};
                end
                [last_sol, infoQ] = mpc.Q.solve(pars, last_sol, true, ...
                                                            multistart);
            end

            % choose if to apply perturbation
            if rand < 0.1 * exp(-(ep - 1) / 5)
                pert = perturb_mag * exp(-(ep - 1) / 5) * randn;
            else
                pert = 0;
            end

            % run V(s(k))
            pars = struct( ...
                'd', D(:, K*(ep-1) + k:K*(ep-1) + k + M*Np-1), ...
                'w0', w, 'rho0', rho, 'v0', v, ...
                'r_last', r, 'perturbation', pert);
            if ~approx.Veq
                pars.a = a;
            end
            for n = fieldnames(rl.pars)'
                pars.(n{1}) = rl.pars.(n{1}){end};
            end
            [last_sol, infoV] = mpc.V.solve(pars,last_sol,true,multistart);

            % save to memory if successful, or log error 
            if ep > 1 || k_mpc > 1
                if infoV.success && infoQ.success
                    % compute td error
                    td_err = full(Lrl(w_prev, rho_prev, v_prev)) ...
                                        + discount * infoV.f  - infoQ.f;
                    td_error{ep}(k_mpc) = td_err;
                    td_error_perc{ep}(k_mpc) = td_err / infoQ.f;

                    % compute numerical gradients w.r.t. params
                    dQ = infoQ.get_value(deriv.Q.dL);
                    d2Q = infoQ.get_value(deriv.Q.d2L);

                    % store in memory
                    replaymem.add(struct( ...
                                    'A', td_err * d2Q - dQ * dQ', ...
                                    'b', td_err * dQ, 'dQ', dQ));

                    % util.info(toc(start_tot_time), ep, ...
                    %                     toc(start_ep_time), t(k), k, K);
                else
                    nb_fail = nb_fail + 1;
                    msg = '';
                    if ~infoV.success
                        msg = append(msg, sprintf('V: %s. ', infoV.msg));
                        
                    end
                    if ~infoQ.success
                        msg = append(msg, sprintf('Q: %s.', infoQ.msg));
                    end
                    util.info(toc(start_tot_time), ep, ...
                                    toc(start_ep_time), t(k), k, K, msg);
                end
            end

            % get optimal a_k from V(s_k)
            r = last_sol.r(:, 1);

            % save slack variables
            for n = slacknames
                slacks.(n){ep}(:, k_mpc) = last_sol.(n)(:);
            end

            % save for next transition
            r_prev_prev = r_prev;
            r_prev = r;
            w_prev = w;
            rho_prev = rho;
            v_prev = v;
        end

        % step state (according to the true dynamics)
        [q_o, w_next, q, rho_next, v_next] = dynamics.real.f(...
            w, rho, v, r, D(:, k), ...
            true_pars.rho_crit, true_pars.a, true_pars.v_free);

        % save current state and other infos
        origins.demand{ep}(:, k) = D(:, k);
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
        if mod(k, rl_update_freq) == 0 && ep > 1
            % sample batch 
            sample = replaymem.sample(rl_mem_sample, rl_mem_last);
            
            % compute hessian and descent direction
            f = lr * rlmpc.modify_hessian(sample.A) \ sample.b;
            
            % perform constrained update
            rl.pars = rlmpc.rl_constrained_update(rl.pars, rl.bounds, f);

            % log update result
            msg = sprintf('RL update %i with %i samples: ', ...
                length(rl.pars.rho_crit) - 1, sample.n);
            for name = fieldnames(rl.pars)'
                msg = append(msg, name{1}, '=', ...
                            mat2str(rl.pars.(name{1}){end}(:)', 6), '; ');
            end
            util.info(toc(start_tot_time), ep, toc(start_ep_time), ...
                                                        t(k), k, K, msg);
        end
    end
    exec_times(ep) = toc(start_ep_time);

    % save every episode in a while (exclude some variables)
    if mod(ep - 1, save_freq) == 0
        warning('off');
        save('checkpoint', '-regexp', ...
            '^(?!(mpc|cost|ctrl|dQlagr|r_last|v_free_tr|weight_)).*')
        warning('on');
        util.info(toc(start_tot_time), ep, exec_times(ep), t(end), K, ...
            K, 'checkpoint saved');
    end

    % log intermediate results
    ep_Jtot = full(sum( ...
            Lrl(origins.queue{ep}, links.density{ep}, links.speed{ep})));
    ep_TTS = full(sum(TTS(origins.queue{ep}, links.density{ep})));
    util.info(toc(start_tot_time), ep, exec_times(ep), t(end), K, K, ...
        sprintf('episode %i: Jtot=%.3f, TTS=%.3f, fails=%i(%.1f%%)', ...
        ep, ep_Jtot, ep_TTS, nb_fail, nb_fail / K * M * 100));

    % plot performance
    if ~exist('ph_J', 'var') || ~isvalid(ph_J)
        figure;
        yyaxis left, 
        ph_J = plot(ep, ep_Jtot, '-o');
        ylabel('J')
        yyaxis right, 
        ph_TTS = plot(ep, ep_TTS, '-o');
        ylabel('TTS')
    else
        set(ph_J, 'XData', [ph_J.XData, ep]);
        set(ph_J, 'YData', [ph_J.YData, ep_Jtot]);
        set(ph_TTS, 'XData', [ph_TTS.XData, ep]);
        set(ph_TTS, 'YData', [ph_TTS.YData, ep_TTS]);
    end
    drawnow;
end
exec_time_tot = toc(start_tot_time);
diary off



%% Saving and plotting
% build arrays
rl.pars = structfun(@(x) cell2mat(x), rl.pars, 'UniformOutput', false);

% clear and save
clear ans args cost ctrl D d d_cong deriv dtd_err d1 d2 dQ d2Q dynamics ...
      exp ep_Jtot ep_TTS f filter_num filter_den H i infoQ infoV Is k ...
      Lagr last_sol load_checkpoint log_filename methods mpc msg n ...
      nb_fail name pars penalty ph_J ph_TTS q q_o r r_first r_last r_prev ...
      rcondA replaymem rho rho_next rho_prev save_freq start_ep_time ...
      start_ep start_tot_time td_err sz v v_next v_prev ...
      w w_next w_prev
delete checkpoint.mat
save(strcat(runname, '_data.mat'));

% plot
run visualization.m
