% made with MATLAB 2021b
clc, clearvars, close all, diary off, warning('on')
rng(42)
runname = datestr(datetime, 'yyyymmdd_HHMMSS');
save_freq = 2;                          % checkpoint saving frequency



%% Instanciate environment
episodes = 75;                          % number of episodes to repeat
[sim, mdl, mpc] = util.get_pars();

% create gym 
env = metanet.TrafficEnv(episodes, sim, mdl);
env.reset();

% create known (wrong) model parameters
wrong_pars = struct('a', env.model.a * 1.3, ...
                    'v_free', env.model.v_free * 1.3, ...
                    'rho_crit', env.model.rho_crit * 0.7);

% TODO: delete these
% plot((0:length(D) - 1) * T, (D .* [1; 1; 50])'),
% legend('O1', 'O2', 'cong_{\times50}'), xlabel('time (h)'), ylabel('demand (h)')
% ylim([0, 4000])



%% MPC-based RL
% TODO: move all of this to some get_agent method (also costs above)

% cost terms
[TTS, Rate_var] = metanet.get_mpc_costs(mdl, sim, mpc);
[Vcost,Lcost,Tcost] = rlmpc.get_mpc_costs(mdl, mpc,'affine','diag','diag');

% TODO: move this to env, need to remove depence on mpc
Lrl = rlmpc.get_rl_cost(mdl, mpc, TTS, Rate_var); 

% build mpc-based value function approximators

for name = ["Q", "V"]
    % instantiate an MPC
    ctrl = rlmpc.NMPC(name, mdl, sim, mpc, env.dynamics);

    % grab the names of the slack variables
    % TODO: get rid of these names
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
                 ctrl.pars.weight_V);
    for k = 2:mpc.M * mpc.Np
        cost = cost + Lcost(ctrl.vars.rho(:, k), ...
                            ctrl.vars.v(:, k), ...
                            ctrl.pars.rho_crit, ...
                            ctrl.pars.v_free_tracking, ... % ctrl.pars.v_free
                            ctrl.pars.weight_L);
    end
    % terminal cost
    cost = cost + Tcost(ctrl.vars.rho(:, end), ...
                        ctrl.vars.v(:, end), ...
                        ctrl.pars.rho_crit, ...
                        ctrl.pars.v_free_tracking, ...
                        ctrl.pars.weight_T);
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
                            ctrl.vars.r(:, 1) ./ mpc.normalization.r);
    end

    % save to struct
    mpc.(name) = ctrl;
end
clear ctrl



%% Simulation
% initial conditions
% TODO: can we remove these?
r_prev = env.r_prev;

% initial learnable Q/V function approxiamtion weights and their bounds
args = cell(0, 3);
args(end + 1, :) = {'rho_crit', {wrong_pars.rho_crit}, [10, mdl.rho_max * 0.9]};
args(end + 1, :) = {'v_free', {wrong_pars.v_free}, [30, 300]};
args(end + 1, :) = {'v_free_tracking', {wrong_pars.v_free}, [30, 300]};
args(end + 1, :) = {'weight_V', ...
                        {ones(size(mpc.V.pars.weight_V))}, [-inf, inf]};
args(end + 1, :) = {'weight_L', ...
                        {ones(size(mpc.V.pars.weight_L))}, [0, inf]};
args(end + 1, :) = {'weight_T', ...
                        {ones(size(mpc.V.pars.weight_T))}, [0, inf]};
args(end + 1, :) = {'weight_rate_var', {mpc.rate_var_penalty}, [1e-3, 1e3]};
if isfield(mpc.V.vars, 'slack_w_max')
    args(end+1,:) = {'weight_slack_w_max', ...
        {ones(size(mpc.V.pars.weight_slack_w_max)) * ...
                                        mpc.con_violation_penalty}, [0, inf]};
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
for n = ["Q", "V"]
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
    'w', repmat(env.state.w, 1, mpc.M * mpc.Np + 1), ...
    'rho', repmat(env.state.rho, 1, mpc.M * mpc.Np + 1), ...
    'v', repmat(env.state.v, 1, mpc.M * mpc.Np + 1), ...
    'r', repmat(env.r_prev, 1, mpc.Nc));
for n = slacknames
    last_sol.(n) = zeros(size(mpc.V.vars.(n)));
end

% initialize constrained QP RL update maximum lagrangian multiplier
% lam_inf = 1;

% initialize mpc solvers
mpc.Q.init_solver(mpc.opts_ipopt);
mpc.V.init_solver(mpc.opts_ipopt);

% create replay memory
replaymem = rlmpc.ReplayMem(mpc.mem_cap, 'none', 'td_err', 'dQ', ...
                            'd2Q', 'target', 'solQ', 'parsQ', 'last_solQ');

% start logging
diary(strcat(runname, '_log.txt'))
fprintf(['# Fields: [Realtime_tot|Episode_n|Realtime_episode] ', ...
    '- [Sim_time|Sim_iter|Sim_perc] - Message\n'])

% TODO: to be removed
K = sim.K;
M = mpc.M;
Np = mpc.Np;

start_tot_time = tic;
for ep = 1:episodes
    % preallocate episode result containers
    % TODO: let the monitor class instance save all these variables
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
        k = env.k;
        if mod(k, M) == 1
            k_mpc = ceil(k / M); % mpc iteration

            % run Q(s(k-1), a(k-1)) (condition excludes very first iteration)
            if ep > 1 || k_mpc > 1
                pars = struct( ...
                    'd', env.demand(:, K*(ep-1) + k-M:K*(ep-1) + k-M + M*Np-1), ...
                    'w0', state_prev.w, 'rho0', state_prev.rho, 'v0', state_prev.v, ...
                    'a', wrong_pars.a, 'r_last', r_prev_prev, 'r0', r_prev); % a(k-2) and a(k-1)
                for n = fieldnames(rl.pars)'
                    pars.(n{1}) = rl.pars.(n{1}){end};
                end
                [last_sol, infoQ] = mpc.Q.solve(pars, last_sol, true, ...
                                                            mpc.multistart);
                parsQ = pars;           % to be saved for backtracking
                last_solQ = last_sol;   % to be saved for backtracking
            end

            % choose if to apply perturbation
            if rand < 0.1 * exp(-(ep - 1) / 5)
                pert = mpc.perturb_mag * exp(-(ep - 1) / 5) * randn;
            else
                pert = 0;
            end

            % run V(s(k))
            pars = struct( ...
                'd', env.demand(:, K*(ep-1) + k:K*(ep-1) + k + M*Np-1), ...
                'w0', env.state.w, 'rho0', env.state.rho, 'v0', env.state.v, 'a', wrong_pars.a, ...
                'r_last', env.r_prev, 'perturbation', pert);
            for n = fieldnames(rl.pars)'
                pars.(n{1}) = rl.pars.(n{1}){end};
            end
            [last_sol, infoV] = mpc.V.solve(pars, last_sol, true, mpc.multistart);

            % save to memory if successful, or log error 
            if ep > 1 || k_mpc > 5 % skip the first td errors to let them settle
                if infoV.success && infoQ.success
                    % compute td error
                    target = full(Lrl(state_prev.w, state_prev.rho, state_prev.v, ...
                                      r_prev, r_prev_prev)) ...
                                + mpc.discount * infoV.f;
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
                                    toc(start_ep_time), sim.t(k), k, K, msg);
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
            state_prev = env.state;
        end

        % save current state, input, and demand
        origins.queue{ep}(:, k) = env.state.w;
        links.density{ep}(:, k) = env.state.rho;
        origins.rate{ep}(:, k) = r;
        links.speed{ep}(:, k) = env.state.v;
        origins.demand{ep}(:, k) = env.d;

        % step the environment
        [~, ~, done, info] = env.step(r);

        % save current flows
        origins.flow{ep}(:, k) = info.q_o;
        links.flow{ep}(:, k) = info.q;

        % perform RL updates
        if mod(k, mpc.update_freq) == 0 && ep > 1
            % sample batch
            sample = replaymem.sample(mpc.mem_sample, mpc.mem_last);

            % compute descent direction (for the Gauss-Newton just dQdQ')
            g = -sample.dQ * sample.td_err;
            H = sample.dQ * sample.dQ' - ...
                sum(sample.d2Q .* reshape(sample.td_err, 1, 1, []), 3);
            % H = [];
            [p, H_mod] = rlmpc.descent_direction(g, H, 1);

            % lr (fixed or backtracking)
            lr_ = mpc.lr0;
            % lr_ = rlmpc.constr_backtracking(mpc.Q, deriv.Q, p, sample, rl, mpc.pars.max_delta, lam_inf);

            % perform constrained update and save its maximum multiplier
            [rl.pars, ~, lam] = rlmpc.constr_update(rl.pars, rl.bounds, ...
                                              lr_ * p, mpc.max_delta);
            % lam_inf = max(lam_inf, lam);

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
                                                        sim.t(k), k, K, msg);
        end
    end
    exec_times(ep) = toc(start_ep_time);

    % save every episode in a while (exclude some variables)
    if mod(ep - 1, save_freq) == 0
        warning('off');
        save('checkpoint')
        warning('on');
        util.info(toc(start_tot_time), ep, exec_times(ep), sim.t(end), K, ...
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
    constr_viol = sum(origins.queue{ep}(2, :) > mdl.max_queue) / K;
    util.info(toc(start_tot_time), ep, exec_times(ep), sim.t(end), K, K, ...
        sprintf('episode %i: Jtot=%.3f, TTS=%.3f, fails=%i(%.1f%%)', ...
        ep, ep_Jtot, ep_TTS, nb_fail, nb_fail / K * M * 100));

    % plot performance
    if ~exist('ph_J', 'var') || ~isvalid(ph_J)
        figure; tiledlayout(4, 1);
        nexttile; 
        yyaxis left, 
        ph_J = semilogy(ep_Jtot, '-o');
        ylabel('J(\pi)')
        yyaxis right, 
        ph_TTS = plot(ep_TTS, '-o');
        xlabel('episode'), ylabel('TTS(\pi)'),
        nexttile; 
        ph_g_norm = semilogy(g_norm_avg, '-*');
        xlabel('episode'), ylabel('average ||g||')
        nexttile; 
        ph_p_norm = semilogy(1, '-o');
        xlabel('update'), ylabel('||p||'), 
        nexttile; 
        ph_viol = area(constr_viol);
        xlabel('episode'), ylabel('constr. violation'), 
    else
        set(ph_J, 'YData', [ph_J.YData, ep_Jtot]);
        set(ph_TTS, 'YData', [ph_TTS.YData, ep_TTS]);
        set(ph_g_norm, 'YData', [ph_g_norm.YData, g_norm_avg]);
        set(ph_p_norm, 'YData', p_norm);
        set(ph_viol, 'YData', [ph_viol.YData, constr_viol]);
    end
    drawnow;
end
exec_time_tot = toc(start_tot_time);
diary off



%% Saving and plotting
delete checkpoint.mat
save(strcat(runname, '_data.mat'));

% plot
run visualization.m
