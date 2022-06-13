% made with MATLAB 2021b
clc, clearvars, close all, diary off, warning('on')
rng(42)
runname = datestr(datetime, 'yyyymmdd_HHMMSS');
load_checkpoint = false;



%% Model
% simulation
episodes = 75;                          % number of episodes to repeat
Tfin = 2;                               % simulation time per episode (h)
T = 10 / 3600;                          % simulation step size (h)
K = Tfin / T;                           % simulation steps per episode
t = (0:(K - 1)) * T;                    % time vector (h)

% model parameters
mdl = metanet.get_pars();

% known (wrong) model parameters
a = mdl.a * 1.3;
v_free = mdl.v_free * 1.3;
rho_crit = mdl.rho_crit * 0.7;


% TODO: remove this garbage
max_queue = [Inf, mdl.max_queue];       % maximum queue (veh) constraint






%% Disturbances
D = util.get_demand_profiles(t, episodes + 1, 'fixed'); % +1 to avoid out-of-bound access
assert(size(D, 1) == mdl.n_dist, 'mismatch found')

% plot((0:length(D) - 1) * T, (D .* [1; 1; 50])'),
% legend('O1', 'O2', 'cong_{\times50}'), xlabel('time (h)'), ylabel('demand (h)')
% ylim([0, 4000])



%% MPC-based RL
% TODO: put all these parameters in rlmpc.get_pars()
% parameters (constant)
max_in_and_out = [false, true];         % whether to apply max to inputs and outputs of dynamics
%
Np = 4;                                 % prediction horizon - \approx 3*L/(M*T*v_avg)
Nc = 3;                                 % control horizon
M  = 6;                                 % horizon spacing factor
eps = 0;                                % nonnegative constraint precision
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
perturb_mag = 100;                      % magnitude of exploratory perturbation
rate_var_penalty = 4e-2;
methods = {'ipopt', 'sqpmethod', 'fmincon'};
method = methods{1};                    % solver method for MPC
multistart = 1; %4 * 4;                 % multistarting NMPC solver
soft_domain_constraints = false;        % whether to use soft constraints on positivity of states (either this, or max on output)
if ~soft_domain_constraints && ~max_in_and_out(2)
    warning('Dynamics can be negative and hard constraints unfeasible')
end
%
discount = 0.99;                        % rl discount factor
lr = 1e-3;                              % fixed rl learning rate (no line search)
grad_desc_version = 1;                  % type of gradient descent/hessian modification (1 is the only one working out)
max_delta = 1 / 5;                      % percentage of maximum parameter change in a single update
con_violation_penalty = 10;             % penalty for constraint violations
rl_update_freq = K / 2;                 % when rl should update
rl_mem_cap = K / M * 10;                % RL experience replay capacity
rl_mem_sample = K / M * 5;              % RL experience replay sampling size
rl_mem_last = K / M;                    % percentage of last experiences to include in sample
save_freq = 2;                          % checkpoint saving frequency

% create a symbolic casadi function for the dynamics (both true and nominal)
dynamics = metanet.get_dynamics(mdl, T);

% cost terms
[TTS, Rate_var] = metanet.get_mpc_costs(mdl, Nc, T);
[Vcost, Lcost, Tcost] = rlmpc.get_mpc_costs(mdl, 'affine', 'diag', 'diag');
Lrl = rlmpc.get_rl_cost(mdl, TTS, Rate_var, rate_var_penalty, ...
                                                    con_violation_penalty);

% build mpc-based value function approximators
mpc = struct;
for name = ["Q", "V"]
    % instantiate an MPC
    ctrl = rlmpc.NMPC(name, Np, Nc, M, dynamics, max_queue, ...
                           soft_domain_constraints, eps, ...
                           true, mdl.rho_max, mdl.C, T);

    % normalization constants
    normalization.w = mdl.max_queue;
    normalization.rho = mdl.rho_max;
    normalization.v = v_free;
    normalization.r = ctrl.r_bnd{2};

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
        % w_max slacks are punished less because the weight is learnable
        if endsWith(n, 'w_max')
            cost = cost ... 
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

    % case-specific modification
    if strcmp(name, "Q")
        % Q approximator has additional constraint on first action
        ctrl.add_par('r0', [size(ctrl.vars.r, 1), 1]);
        ctrl.add_con('r0_blocked', ctrl.vars.r(:, 1) - ctrl.pars.r0, 0, 0);
    elseif strcmp(name, "V")
        % V approximator has perturbation to enhance exploration
        ctrl.add_par('perturbation', size(ctrl.vars.r(:, 1)));
        ctrl.minimize(ctrl.f + ctrl.pars.perturbation' * ...
                                    ctrl.vars.r(:, 1) ./ normalization.r);
    end

    % save to struct
    mpc.(name) = ctrl;
end
clear ctrl



%% Simulation
% initial conditions
r = mpc.V.r_bnd{2};                     % metering rate/flow
r_prev = r;                             % previous rate
[w, rho, v] = util.steady_state(dynamics.f, ...
    zeros(mdl.n_origins, 1), 10 * ones(mdl.n_links, 1), 100 * ones(mdl.n_links, 1), ...
    r, D(:, 1), mdl.rho_crit, mdl.a, mdl.v_free);

% initial learnable Q/V function approxiamtion weights and their bounds
args = cell(0, 3);
args(end + 1, :) = {'rho_crit', {rho_crit}, [10, mdl.rho_max * 0.9]};
args(end + 1, :) = {'v_free', {v_free}, [30, 300]};
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
rl_history.lr = {};
rl_history.p_norm = {};
rl_history.H_mod = {};
rl_history.g_norm = cell(1, episodes);
rl_history.td_error = cell(1, episodes);
rl_history.td_error_perc = cell(1, episodes); 

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

% initialize constrained QP RL update maximum lagrangian multiplier
lam_inf = 1;

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
replaymem = rlmpc.ReplayMem(rl_mem_cap, 'none', 'td_err', 'dQ', 'd2Q', ...
                                'target', 'solQ', 'parsQ', 'last_solQ');

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
        slacks.(n){ep} = nan(numel(mpc.V.vars.(n)), K / M);
    end
    rl_history.g_norm{ep} = nan(1, K / M);
    rl_history.td_error{ep} = nan(1, K / M);
    rl_history.td_error_perc{ep} = nan(1, K / M);

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
                    'a', a, 'r_last', r_prev_prev, 'r0', r_prev); % a(k-2) and a(k-1)
                for n = fieldnames(rl.pars)'
                    pars.(n{1}) = rl.pars.(n{1}){end};
                end
                [last_sol, infoQ] = mpc.Q.solve(pars, last_sol, true, ...
                                                            multistart);
                parsQ = pars;           % to be saved for backtracking
                last_solQ = last_sol;   % to be saved for backtracking
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
                'w0', w, 'rho0', rho, 'v0', v, 'a', a, ...
                'r_last', r, 'perturbation', pert);
            for n = fieldnames(rl.pars)'
                pars.(n{1}) = rl.pars.(n{1}){end};
            end
            [last_sol, infoV] = mpc.V.solve(pars,last_sol,true,multistart);

            % save to memory if successful, or log error 
            if ep > 1 || k_mpc > 5 % skip the first td errors to let them settle
                if infoV.success && infoQ.success
                    % compute td error
                    target = full(Lrl(w_prev, rho_prev, v_prev, r_prev, ...
                                                          r_prev_prev)) ...
                                + discount * infoV.f;
                    td_err = target - infoQ.f;

                    % compute numerical gradients w.r.t. params
                    dQ = infoQ.get_value(deriv.Q.dL);
                    d2Q = infoQ.get_value(deriv.Q.d2L);

                    % store in memory
                    replaymem.add('td_err', td_err, 'dQ', dQ, ...
                        'd2Q', d2Q, 'target', target, 'solQ', infoQ.f, ...
                        'parsQ', parsQ, 'last_solQ', last_solQ);

                    % save stuff
                    rl_history.g_norm{ep}(k_mpc) = norm(td_err * dQ, 2);
                    rl_history.td_error{ep}(k_mpc) = td_err;
                    rl_history.td_error_perc{ep}(k_mpc) = td_err / infoQ.f;
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
        [q_o, w_next, q, rho_next, v_next] = dynamics.f(...
            w, rho, v, r, D(:, K*(ep-1) + k), ...
            mdl.rho_crit, mdl.a, mdl.v_free);

        % save current state and other infos
        origins.demand{ep}(:, k) = D(:, K*(ep-1) + k);
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

            % compute descent direction (for the Gauss-Newton just dQdQ')
            g = -sample.dQ * sample.td_err;
            if grad_desc_version ~= 0
                H = sample.dQ * sample.dQ' - ...
                    sum(sample.d2Q .* reshape(sample.td_err, 1, 1, []), 3);
            else
                H = [];
            end
            [p, H_mod] = rlmpc.descent_direction(g, H, grad_desc_version);

            % lr backtracking
            if ~exist('lr', 'var')
                lr_ = rlmpc.constr_backtracking(mpc.Q, deriv.Q, p, ...
                                        sample, rl, max_delta, lam_inf);
            else
                lr_ = lr; % / sample.n;
            end

            % perform constrained update and save its maximum multiplier
            [rl.pars, ~, lam] = rlmpc.constr_update(rl.pars, rl.bounds, ...
                                                    lr_ * p, max_delta);
            % lam_inf = 0.25 * lam + 0.75 * lam_inf; % exp moving average
            lam_inf = max(lam_inf, lam);

            % save stuff
            rl_history.lr{end + 1} = lr_;
            rl_history.H_mod{end + 1} = H_mod;
            rl_history.p_norm{end + 1} = norm(p);

            % log update result
            msg = sprintf('update %i (N=%i, lr=%1.3e, Hmod=%1.3e): ', ...
                          length(rl.pars.rho_crit) - 1, sample.n, ...
                          rl_history.lr{end}, rl_history.H_mod{end});
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
    if ep == 1
        rate_prev = [origins.rate{ep}(1), origins.rate{ep}(1:end-1)];
    else
        rate_prev = [origins.rate{ep-1}(end), origins.rate{ep}(1:end-1)];
    end
    ep_Jtot = full(sum(Lrl(origins.queue{ep}, links.density{ep}, ...
                           links.speed{ep}, origins.rate{ep}, rate_prev)));
    ep_TTS = full(sum(TTS(origins.queue{ep}, links.density{ep})));
    g_norm_avg = mean(rl_history.g_norm{ep}, 'omitnan');
    p_norm = cell2mat(rl_history.p_norm);
    util.info(toc(start_tot_time), ep, exec_times(ep), t(end), K, K, ...
        sprintf('episode %i: Jtot=%.3f, TTS=%.3f, fails=%i(%.1f%%)', ...
        ep, ep_Jtot, ep_TTS, nb_fail, nb_fail / K * M * 100));

    % plot performance
    if ~exist('ph_J', 'var') || ~isvalid(ph_J)
        figure; tiledlayout(3, 1);
        nexttile; 
        yyaxis left, 
        ph_J = semilogy(ep, ep_Jtot, '-o');
        ylabel('J(\pi)')
        yyaxis right, 
        ph_TTS = plot(ep, ep_TTS, '-o');
        xlabel('episode'), ylabel('TTS(\pi)'),
        nexttile; 
        ph_g_norm = semilogy(ep, g_norm_avg, '-*');
        xlabel('episode'), ylabel('average ||g||')
        nexttile; 
        ph_p_norm = semilogy(1, '-o');
        xlabel('update'), ylabel('||p||'), 
    else
        set(ph_J, 'XData', [ph_J.XData, ep]);
        set(ph_J, 'YData', [ph_J.YData, ep_Jtot]);
        set(ph_TTS, 'XData', [ph_TTS.XData, ep]);
        set(ph_TTS, 'YData', [ph_TTS.YData, ep_TTS]);
        set(ph_g_norm, 'XData', [ph_g_norm.XData, ep]);
        set(ph_g_norm, 'YData', [ph_g_norm.YData, g_norm_avg]);
        set(ph_p_norm, 'YData', p_norm);
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
