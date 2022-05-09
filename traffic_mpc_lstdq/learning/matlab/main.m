% made with MATLAB 2021b
clc, clearvars, close all, diary off
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

% network size
n_origins = 2;
n_links = 3;
control_origin_ramp = false;    % toggle this to control origin ramp
n_ramps = 1 + control_origin_ramp;                   

% segments
L = 1;                          % length of links (km)
lanes = 2;                      % lanes per link (adim)

% origins O1 and O2 
C = [3500, 2000];               % on-ramp capacity (veh/h/lane)
max_queue = [150, 50];          % maximum queue (veh) - for constraint
if ~control_origin_ramp
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
Np = 7;                             % prediction horizon
Nc = 3;                             % control horizon
M = 6;                              % horizon spacing factor
eps = 1e-4;                         % nonnegative constraint precision
plugin_opts = struct('expand', true, 'print_time', false);
solver_opts = struct('print_level', 0, 'max_iter', 3e3, 'tol', 1e-7, ...
    'barrier_tol_factor', 1e-3);
perturb_mag = 0;                    % magnitude of exploratory perturbation
rate_var_penalty = 0.4;             % penalty weight for rate variability
discount = 1;                       % rl discount factor
lr = 1e-4;                          % rl learning rate
con_violation_penalty = 10;         % rl penalty for constraint violations
rl_update_freq = round(K / 5);      % when rl should update
rl_mem_cap = rl_update_freq * 5;    % RL experience replay capacity
rl_mem_sample = rl_update_freq * 2; % RL experience replay sampling size
rl_mem_last = 0.5;                  % percentage of last experiences to include in sample
save_freq = 2;                      % checkpoint saving frequency

% create a symbolic casadi function for the dynamics
n_dist = size(D, 1);                
F = metanet.get_dynamics(n_links, n_origins, n_ramps, n_dist, ...
    T, L, lanes, C, rho_max, tau, delta, eta, kappa, eps);

% cost terms (learnable MPC, metanet and RL)
[Vcost, Lcost, Tcost] = rlmpc.get_mpc_costs(n_links, n_origins, ...
    'affine', 'diag', 'diag');    
TTS = metanet.TTS(n_links, n_origins, T, L, lanes);
Rate_var = metanet.rate_variability(n_ramps, Nc);
Lrl = rlmpc.get_rl_cost(n_links, n_origins, n_ramps, TTS, max_queue, ...
    con_violation_penalty);
normalization.w = max(max_queue(~isinf(max_queue)));   % normalization constants
normalization.rho = rho_max;
normalization.v = v_free; 

% build mpc-based value function approximators
mpc = struct;
for name = ["Q", "V"]
    % instantiate an MPC
    ctrl = rlmpc.NMPC(Np, Nc, M, F, eps);
    ctrl.opti.solver('ipopt', plugin_opts, solver_opts);

    % set soft constraint on ramp queue
    slack = ctrl.add_var('slack', n_ramps, M * Np + 1);
    ctrl.opti.subject_to(-slack(:) + eps^2 <= 0);
    if ~control_origin_ramp
        ctrl.opti.subject_to( ...
            ctrl.vars.w(2, :) - slack(1, :) - max_queue(2) <= 0)
    else
        ctrl.opti.subject_to( ...
            ctrl.vars.w(1, :) - slack(1, :) - max_queue(1) <= 0)
        ctrl.opti.subject_to( ...
            ctrl.vars.w(2, :) - slack(2, :) - max_queue(2) <= 0)
        warning('there might be problems with QP update parameter size');
    end

    % create required parameters
    v_free_tracking = ctrl.add_par('v_free_tracking', 1, 1); 
    weight_V = ctrl.add_par('weight_V', size(Vcost.mx_in(3)));
    weight_L = ctrl.add_par('weight_L', size(Lcost.mx_in(4)));
    weight_T = ctrl.add_par('weight_T', size(Tcost.mx_in(2)));
    weight_slack = ctrl.add_par('weight_slack', flip(size(slack)));
    weight_rate_var = ctrl.add_par('weight_rate_var', 1, 1);
    r_last = ctrl.add_par('r_last', size(ctrl.vars.r, 1), 1);

    % compute the actual symbolic cost expression with opti vars and pars
    cost = Vcost(...                                        % affine/constant initial cost
            ctrl.vars.w(:, 1), ...     
            ctrl.vars.rho(:, 1), ...            
            ctrl.vars.v(:, 1), ...
            weight_V, ...
            normalization.w,...
            normalization.rho,...
            normalization.v) ...                              
        + Tcost(...                                         % quadratic terminal cost
            ctrl.vars.rho(:, end), ...    
            ctrl.pars.rho_crit, ...
            weight_T, ...
            normalization.rho) ...
        + weight_slack(end, :) * slack(:, end) ...          % terminal slack penalty
        + Rate_var(r_last, ctrl.vars.r) * weight_rate_var;  % terminal rate variability
    % stage terms
    for k = 1:M * Np
        cost = cost ...
            + weight_slack(k, :) * slack(:, k) ...          % slack penalty
            + Lcost(...                                     % quadratic stage cost
                ctrl.vars.rho(:, k), ...         
                ctrl.vars.v(:, k), ...           
                ctrl.pars.rho_crit, ...
                v_free_tracking, ... % ctrl.pars.v_free, ...
                weight_L, ...
                normalization.rho, ...
                normalization.v) ...                         
            + TTS(ctrl.vars.w(:, k), ctrl.vars.rho(:, k));  % TTS 
    end
    % assign cost to opti
    ctrl.opti.minimize(cost);

    % save to struct
    mpc.(name) = ctrl;
end

% Q approximator has additional constraint on first action
mpc.Q.add_par('r_first', size(mpc.Q.vars.r, 1), 1);
mpc.Q.opti.subject_to(mpc.Q.vars.r(:, 1) - mpc.Q.pars.r_first == 0);

% V approximator has perturbation to enhance exploration
mpc.V.add_par('perturbation', size(mpc.V.vars.r(:, 1)));
mpc.V.opti.minimize( ...
    mpc.V.opti.f + mpc.V.pars.perturbation' * mpc.V.vars.r(:, 1));


%% Simulation
% initial conditions
r = ones(n_ramps, 1);                   % ramp metering rate
r_prev = ones(n_ramps, 1);              % previous rate
[w, rho, v] = util.steady_state(F, ...  % queue, density, speed at steady-state
    [0; 0], [5; 5; 18], [100; 198; 90], r, D(:, 1), ...
    true_pars.a, true_pars.v_free, true_pars.rho_crit);

% initial function approx weights
rl_pars.v_free = {v_free};
rl_pars.rho_crit = {rho_crit};
rl_pars.v_free_tracking = {v_free};
rl_pars.weight_V = {ones(size(weight_V))};
rl_pars.weight_L = {ones(size(weight_L))};
rl_pars.weight_T = {ones(size(weight_T))};
rl_pars.weight_rate_var = {rate_var_penalty};
rl_pars.weight_slack = {ones(size(weight_slack)) * con_violation_penalty};

% rl parameters bounds
rl_pars_bounds.v_free = [30, 300]; 
rl_pars_bounds.rho_crit = [10, 200];
rl_pars_bounds.v_free_tracking = [30, 300];
rl_pars_bounds.weight_V = [-inf, inf];
rl_pars_bounds.weight_L = [0, inf];
rl_pars_bounds.weight_T = [0, inf];
rl_pars_bounds.weight_rate_var = [1e-3, 1e2];
rl_pars_bounds.weight_slack = [0, inf];

% compute symbolic derivatives
deriv = struct;
for n = string(fieldnames(mpc)')
    % assemble all RL parameters in a single vector for V and Q
    deriv.(n).rl_pars = mpc.(n).concat_pars(fieldnames(rl_pars));

    % compute derivative of Lagrangian
    Lagr = mpc.(n).opti.f + mpc.(n).opti.lam_g' * mpc.(n).opti.g;
    deriv.(n).dL = simplify(jacobian(Lagr, deriv.(n).rl_pars)');
    deriv.(n).d2L = simplify(hessian(Lagr, deriv.(n).rl_pars));
end

% preallocate containers for miscellaneous quantities
slack = cell(1, episodes);
td_error = cell(1, episodes);
td_error_perc = cell(1, episodes);  % td error as a percentage of the Q function
exec_times = nan(1, episodes);

% preallocate containers for traffic data (don't use struct(var, cells))
origins.queue = cell(1, episodes);
origins.flow = cell(1, episodes);
origins.rate = cell(1, episodes);
origins.demand = cell(1, episodes);
links.flow = cell(1, episodes);
links.density = cell(1, episodes);
links.speed = cell(1, episodes);

% initialize mpc last solutions
last_sol = struct( ...
    'w', repmat(w, 1, M * Np + 1), ...
    'r', ones(size(mpc.V.vars.r)), ...
    'rho', repmat(rho, 1, M * Np + 1), ...
    'v', repmat(v, 1, M * Np + 1), ...
    'slack', ones(size(mpc.V.vars.slack)) * eps^2);

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
    slack{ep} = nan(size(mpc.V.vars.slack, 1), ceil(K / M));
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
                pars = struct(...
                    'd', D(:, K*(ep-1) + k-M:K*(ep-1) + k-M + M*Np-1), ...
                    'w0', w_prev, 'rho0', rho_prev, 'v0', v_prev, ...
                    'a', a, 'v_free', rl_pars.v_free{end}, ...
                    'rho_crit', rl_pars.rho_crit{end}, ...
                    'v_free_tracking', rl_pars.v_free_tracking{end}, ...
                    'weight_V', rl_pars.weight_V{end}, ...
                    'weight_L', rl_pars.weight_L{end}, ...
                    'weight_T', rl_pars.weight_T{end}, ...
                    'weight_slack', rl_pars.weight_slack{end}, ...
                    'weight_rate_var', rl_pars.weight_rate_var{end}, ...
                    'r_last', r_prev_prev, 'r_first', r_prev); % a(k-2) and a(k-1)
                last_sol.slack = ones(size(mpc.Q.vars.slack)) * eps^2;
                [last_sol, info_Q] = mpc.Q.solve(pars, last_sol);
            end

            % choose if to apply perturbation
            if rand < 0.5 * exp(-(ep - 1) / 5)
                pert = perturb_mag * exp(-(ep - 1) / 5) * randn;
            else
                pert = 0;
            end

            % run V(s(k))
            pars = struct(...
                'd', D(:, K*(ep-1) + k:K*(ep-1) + k + M*Np-1), ...
                'w0', w, 'rho0', rho, 'v0', v, 'a', a, ...
                'v_free', rl_pars.v_free{end}, ...
                'rho_crit', rl_pars.rho_crit{end}, ...
                'v_free_tracking', rl_pars.v_free_tracking{end}, ...
                'weight_V', rl_pars.weight_V{end}, ...
                'weight_L', rl_pars.weight_L{end}, ...
                'weight_T', rl_pars.weight_T{end}, ...
                'weight_slack', rl_pars.weight_slack{end}, ...
                'weight_rate_var', rl_pars.weight_rate_var{end}, ...
                'r_last', r, 'perturbation', pert);
            last_sol.slack = ones(size(mpc.V.vars.slack)) * eps^2;
            [last_sol, info_V] = mpc.V.solve(pars, last_sol);

            % save to memory if successful, or log error 
            if ep > 1 || k_mpc > 1
                if info_V.success && info_Q.success
                    % compute td error
                    td_err = full(Lrl(w_prev, rho_prev)) + ...
                        discount * info_V.f  - info_Q.f;
                    
                    % save
                    td_error{ep}(k_mpc) = td_err;
                    td_error_perc{ep}(k_mpc) = td_err / info_Q.f;

                    % compute numerical gradients w.r.t. params
                    dQ = info_Q.sol.value(deriv.Q.dL);
                    d2Q = info_Q.sol.value(deriv.Q.d2L);
                    % dV = info_V.sol.value(deriv.V.dL);
                    % dtd_err = discount * dV - dQ;
                    dtd_err = -dQ;

                    % store everything in memory
                    replaymem.add(struct(...
                        'A', td_err * d2Q + dQ * dtd_err', ...
                        'b', td_err * dQ, 'dQ', dQ));

                    util.info(toc(start_tot_time), ep, ...
                        toc(start_ep_time), t(k), k, K);
                else
                    nb_fail = nb_fail + 1;
                    msg = '';
                    if ~info_V.success
                        msg = sprintf('V: %s. ', info_V.error);
                    end
                    if ~info_Q.success
                        msg = append(msg, sprintf('Q: %s.', info_Q.error));
                    end
                    util.info(toc(start_tot_time), ep, ...
                        toc(start_ep_time), t(k), k, K, msg);
                end
            end

            % get optimal a_k from V(s_k)
            r = last_sol.r(:, 1);
            slack{ep}(:, k_mpc) = mean(last_sol.slack, 2);

            % save for next transition
            r_prev_prev = r_prev;
            r_prev = r;
            w_prev = w;
            rho_prev = rho;
            v_prev = v;
        end

        % step state (according to the true dynamics)
        [q_o, w_next, q, rho_next, v_next] = F(...
            w, rho, v, r, D(:, k), ...
            true_pars.a, true_pars.v_free, true_pars.rho_crit);

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
            [sample, Is] = replaymem.sample(rl_mem_sample, rl_mem_last);
            
            % compute hessian and update direction
            A = sample.A + 1e-6 * eye(sample.A);
            rcondA = rcond(A);
            if rcondA > 1e-6
                % 2nd order update
                f = lr * (sample.A \ sample.b);
            else
                warning(['when falling back, might need to ' ...
                    'recalibrate learning rate'])
                msg = sprintf('RL fallback - rcond(A)=%i', rcondA);
                dQ = cell2mat(replaymem.data.dQ(Is));
                A = dQ * dQ' + 1e-6 * eye(size(dQ, 1));
                rcondA = rcond(A);
                if rcond(A) > 1e-6
                    % 2nd order approx update (Gauss-Netwon)
                    f = -lr * (A \ sample.b); 
                else
                    % 1st order update
                    msg = sprintf(' %s, rcond(JQ''JQ)=%f)', msg, rcondA);
                    f = -lr * sample.b / sample.n; 
                end
                util.info(toc(start_tot_time), ep, toc(start_ep_time), ...
                    t(k), k, K, msg);
            end
            H = eye(length(f));
            
            % perform constrained update
            rl_pars = rlmpc.rl_constrained_update( ...
                rl_pars, rl_pars_bounds, H, f);

            % log update result
            msg = sprintf('RL update %i with %i samples: ', ...
                length(rl_pars.v_free) - 1, sample.n);
            for name = fieldnames(rl_pars)'
                if numel(rl_pars.(name{1}){end}) == 1
                    msg = append(msg, name{1}, ...
                        sprintf('=%.3f, ', rl_pars.(name{1}){end}));
                else
                    msg = append(msg, name{1}, '=[', ...
                        num2str(rl_pars.(name{1}){end}(:)', '%.3f '), ...
                        '], ');
                end
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
    ep_Jtot = full(sum(Lrl(origins.queue{ep}, links.density{ep})));
    ep_TTS = full(sum(TTS(origins.queue{ep}, links.density{ep})));
    util.info(toc(start_tot_time), ep, exec_times(ep), t(end), K, K, ...
        sprintf('episode %i: Jtot=%.3f, TTS=%.3f, fails=%i(%.1f%%)', ...
        ep, ep_Jtot, ep_TTS, nb_fail, nb_fail / K * 100));

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
rl_pars = structfun(@(x) cell2mat(x), rl_pars, 'UniformOutput', false);

% clear useless variables
clear cost ctrl d D d1 d2 exp f F filter_num filter_den H i info Is k ...
    last_sol log_filename msg n nb_fail name pars q q_o r r_first  ...
    r_last r_prev rcondA replaymem rho rho_next rho_prev save_freq  ... 
    start_ep_time start_ep start_tot_time td_err sz v v_next v_prev w ...
    w_next w_prev

% save
delete checkpoint.mat
warning('off');
save(strcat(runname, '_data.mat'));
warning('on');
run visualization.m
