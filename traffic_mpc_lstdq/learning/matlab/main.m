% made with MATLAB 2021b
clc, clearvars, close all, diary off, warning('on')
rng(42)
runname = datestr(datetime, 'yyyymmdd_HHMMSS');
save_freq = 2;                          % checkpoint saving frequency



%% Training Environment
iterations = 1;                         % simulation iterations
episodes = 75;                          % number of episodes per iteration
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








%% STILL TO REFACTOR
mpc.Q = agent.Q;
mpc.V = agent.V;

env.reset(575); % TODO: to be moved inside the iteration loop


% TODO: can we remove these?
r_prev = env.env.r_prev;

% compute symbolic derivatives
deriv = struct;
for n = ["Q", "V"]
    % assemble all RL parameters in a single vector for V and Q
    deriv.(n).rl_pars = agent.w_sym(n);

    % compute derivative of Lagrangian
    Lagr = mpc.(n).lagrangian;
    deriv.(n).dL = simplify(jacobian(Lagr, deriv.(n).rl_pars)');
    deriv.(n).d2L = simplify(hessian(Lagr, deriv.(n).rl_pars));
end

% TODO: move these to the agent monitor 
rl_history.lr = {};
rl_history.p_norm = {};
rl_history.H_mod = {};
rl_history.g_norm = cell(1, episodes);
rl_history.td_error = cell(1, episodes);
rl_history.td_error_perc = cell(1, episodes); 

% preallocate containers for miscellaneous quantities
slacks.slack_w_max = cell(1, episodes);

% initialize mpc last solutions to steady-state
last_sol = struct( ...
    'w', repmat(env.state.w, 1, mpc.M * mpc.Np + 1), ...
    'rho', repmat(env.state.rho, 1, mpc.M * mpc.Np + 1), ...
    'v', repmat(env.state.v, 1, mpc.M * mpc.Np + 1), ...
    'r', repmat(env.env.r_prev, 1, mpc.Nc), ...
    'slack_w_max', zeros(size(mpc.V.vars.slack_w_max)));

% initialize constrained QP RL update maximum lagrangian multiplier
% lam_inf = 1;

% initialize mpc solvers
mpc.Q.init_solver(mpc.opts_ipopt);
mpc.V.init_solver(mpc.opts_ipopt);

% create replay memory
replaymem = rlmpc.ReplayMem(mpc.mem_cap, 'none', 'td_err', 'dQ', ...
                            'd2Q', 'target', 'solQ', 'parsQ', 'last_solQ');

% start logging
% diary(strcat(runname, '_log.txt'))
fprintf(['# Fields: [Realtime_tot|Episode_n|Realtime_episode] ', ...
    '- [Sim_time|Sim_iter|Sim_perc] - Message\n'])

% TODO: to be removed
K = sim.K;
M = mpc.M;
Np = mpc.Np;

start_tot_time = tic;
for ep = 1:episodes
    % preallocate episode result containers
    % TODO: let the agent class instance save all these variables
    slacks.slack_w_max{ep} = nan(numel(mpc.V.vars.slack_w_max), K / M);
    rl_history.g_norm{ep} = nan(1, K / M);
    rl_history.td_error{ep} = nan(1, K / M);
    rl_history.td_error_perc{ep} = nan(1, K / M);

    % simulate episode
    start_ep_time = tic;
    mpc.V.failures = 0;
    mpc.Q.failures = 0;
    for k = 1:K
        % check if MPCs must be run
        if mod(k, M) == 1
            k_mpc = ceil(k / M); % mpc iteration

            % run Q(s(k-1), a(k-1)) (condition excludes very first iteration)
            if ep > 1 || k_mpc > 1
                pars = struct( ...
                    'd', env.env.demand(:, K*(ep-1) + k-M:K*(ep-1) + k-M + M*Np-1), ...
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
                'd', env.env.demand(:, K*(ep-1) + k:K*(ep-1) + k + M*Np-1), ...
                'w0', env.state.w, 'rho0', env.state.rho, 'v0', env.state.v, 'a', wrong_pars.a, ...
                'r_last', env.env.r_prev, 'perturbation', pert);
            for n = fieldnames(rl.pars)'
                pars.(n{1}) = rl.pars.(n{1}){end};
            end
            [last_sol, infoV] = mpc.V.solve(pars, last_sol, true, mpc.multistart);

            % save to memory if successful, or log error 
            if ep > 1 || k_mpc > 5 % skip the first td errors to let them settle
                if infoV.success && infoQ.success
                    % compute td error
                    target = L_prev + mpc.discount * infoV.f;
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
                    msg = '';
                    if ~infoV.success
                        msg = append(msg, sprintf('V: %s. ', infoV.msg));

                    end
                    if ~infoQ.success
                        msg = append(msg, sprintf('Q: %s.', infoQ.msg));
                    end
                    util.info(toc(start_tot_time), ep, ...
                                    toc(start_ep_time), sim.t(k), k, K, msg);
                    t1 = toc(start_ep_time);
                    t2 = env.current_ep_exec_time();
                    x___ = 5;
                end
            end

            % get optimal a_k from V(s_k)
            r = last_sol.r(:, 1);

            % save slack variables
            slacks.slack_w_max{ep}(:, k_mpc) = last_sol.slack_w_max(:);

            % save for next transition
            % TODO: remove these, ask the monitor for the state at (-k_mpc)
            r_prev_prev = r_prev;
            r_prev = r;
            state_prev = env.state;
        end

        % step the environment
        [~, L, done, info] = env.step(r);
        if mod(k, M) == 1
            L_prev = L;
        end

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

%     % save every episode in a while (exclude some variables)
%     if mod(ep - 1, save_freq) == 0
%         warning('off');
%         save('checkpoint')
%         warning('on');
%         util.info(toc(start_tot_time), ep, env.exec_times(1, ep), sim.t(end), K, ...
%             K, 'checkpoint saved');
%     end

    % log intermediate results
    % TODO: move printing of episode stats to a method
    ep_Jtot = sum(env.cost.L(1, ep, :));
    ep_TTS = sum(env.cost.TTS(1, ep, :));
    nb_fail = (mpc.V.failures + mpc.Q.failures) / 2;
    util.info(toc(start_tot_time), ep, env.exec_times(1, ep), sim.t(end), K, K, ...
        sprintf('episode %i: J=%.3f, TTS=%.3f, fails=%i(%.1f%%)', ...
        ep, ep_Jtot, ep_TTS, nb_fail, nb_fail / K * M * 100));

    g_norm_avg = mean(rl_history.g_norm{ep}, 'omitnan');
    p_norm = cell2mat(rl_history.p_norm);
    constr_viol = sum(env.origins.queue(1, ep, 2, :) > mdl.max_queue) / K;

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
