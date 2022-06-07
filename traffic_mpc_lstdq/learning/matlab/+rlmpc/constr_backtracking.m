function lr = constr_backtracking(Q, derivQ, p, sample, rl, ...
                                                        lam_inf, k_worst)
    % CONSTR_BACKTRACKING Performs constrained backtracking to find a 
    % learning rate satisfying Wolfe's conditions.
    %
    % Algorithm 3.1 and 3.5 in Nocedal & Wright, Numerical optimization, 2006. 
    % See also: https://nl.mathworks.com/matlabcentral/answers/506524-line-search-algorithm-help
    
    arguments
        Q (1, 1) rlmpc.NMPC
        derivQ (1, 1) struct
        p (:, 1) double
        sample (1, 1) struct
        rl (1, 1) struct
        lam_inf (1, 1) double
        k_worst (1, 1) double = 4
    end

    % pick the worst td error
    [~, worsts] = maxk(sample.td_err, k_worst, 'ComparisonMethod', 'abs');
    pars = sample.parsQ(worsts);
    vals = sample.last_solQ(worsts);
    target = sample.target(worsts);

    % unpack search direction p for each RL parameter
    i = 1;
    p_ = struct;
    for par = fieldnames(rl.pars)'
        sz = length(rl.pars.(par{1}){end});
        p_.(par{1}) = p(i:i+sz-1);
        i = i + sz;
    end

    % shortcut for evaluation of phi - for phi of 0 we don't need to run 
    % the MPC again, since we already have the solution
    eval_phi = @(alpha) evaluate_phi(alpha, p_, target, Q, derivQ, ...
                                     pars, vals, rl, lam_inf * 10);

    % run backtracking line search
    nsteps = 35;
    c1 = 1e-6;
    c2 = 0.999;
    rho = 0.2;
    [phi0, dphi0] = eval_phi(0);
    lr = 1;
    for i = 1:nsteps
        try
            [phi, dphi] = eval_phi(lr);
            if (phi <= phi0 + c1 * lr * dphi0' * p) && ...
                                            (dphi' * p >= c2 * dphi0' * p)
                return
            end
        catch ME
%             warning(ME.identifier, ...
%                         'Error during evaluation of phi: %s', ME.message)
        end
        lr = rho * lr;
    end
    warning('Backtracking line search failed');

%     % run line search algorithm
%     nsteps = 10;
%     c1 = 1e-4;
%     c2 = 0.9;
%     a_prev = 0;
%     a = 1;    
%     [phi0, dphi0] = eval_phi(0);
%     dphi0 = p' * dphi0;
%     phi_prev = phi0;
%     for i = 1:nsteps
%         [phi, dphi] = eval_phi(a);
%         dphi = p' * dphi;
% 
%         if phi > phi0 + c1 * a * dphi0 || (phi > phi_prev && i > 1)
%             zoom.a = [a_prev, a];
%             zoom.phi = [phi_prev, phi];
%         elseif abs(dphi) <= -c2 * dphi0
%             lr = a;
%             return
%         elseif dphi >= 0
%             zoom.a = [a, a_prev]; 
%             zoom.phi = [phi, phi_prev];
%         end
% 
%         if exist('zoom', 'var')
%             for j = 1:nsteps * 5
%                 a = sum(zoom.a) / 2;
%                 [phi, dphi] = eval_phi(a);
%                 if phi > phi0 + c1 * a * dphi0 || phi > zoom.phi(1)
%                     zoom.a(2) = a;
%                     zoom.phi(2) = phi;
%                 else
%                     if abs(dphi) <= -c2 * dphi0
%                         lr = a;
%                         return
%                     elseif dphi * (zoom.a(2) - zoom.a(1)) >= 0
%                         zoom.a(2) = zoom.a(1);
%                         zoom.phi(2) = zoom.phi(1);
%                     end
%                     zoom.a(1) = a;
%                     zoom.phi(1) = phi;
%                 end
%             end
%             lr = a;
%             warning('Line search zoom failed');
%             return
%         end
% 
%         a_prev = a;
%         phi_prev = phi;
%         a = 2 * a;
%     end
%     error('Line search failed');
end


%% local functions
function [phi, dphi]  = evaluate_phi(alpha, p, target, ...
                                            Q, derivQ, pars, vals, rl, v)
    % EVALUATE_PHI. Phi is the following function 
    %               phi(theta) = 1/2N sum_i(target_i - Q(theta))^2 
    %                                   + v * sum_j(max(0, g_j(theta)))
    %   i.e., the lagrangian of the Bellman residuals (not all residuals, 
    %   only a subset). Its derivative is
    %               dphi = -1/N sum_i(td_err_i * dQ) 
    %                            + 1/2 * v * sum_j (dg_j * (sign(g_j) + 1))

    N = length(target);
    rlparnames = string(fieldnames(rl.pars)');
    
    % modify the RL parameters by alpha and assign it 
    % to each parameter struct
    new_rl_pars = struct;
    for par = rlparnames
        new_par = rl.pars.(par){end} + alpha * p.(par);
%         % all parameters expect weight_V must be positive
%         if ~strcmp(par, 'weight_V')
%             new_par = max(1e-3, new_par); 
%         end
        new_rl_pars.(par) = new_par;

        % assign the parameter to the problems' pars structs
        for i = 1:N
            pars(i).(par) = new_par;
        end
    end

    % solve the MPC problem - don't shift values as they are already from 
    % the previously computed solution
    [~, ~, infos] = evalc('Q.solve(pars, vals, false, 1)');
    successes = [infos.success];
    assert(any(successes), 'Evaluation of phi failed')

    % precompute some values
    f = zeros(N, 1);
    dQ = zeros(size(derivQ.dL, 1), N);
    for i = find(successes) % failures remain at zero
        f(i) = infos(i).f;
        dQ(:, i) = infos(i).get_value(derivQ.dL);
    end

    % compute the value of phi and its derivative - normalize by the 
    % number of successes
    phi = 0.5 * sum((target - f).^2) / sum(successes);
    dphi = -dQ * (target - f) / sum(successes);

    % add to phi and dphi the cost related to violation of constraints
    for par = rlparnames
        g_lb = rl.bounds.(par)(1) - new_rl_pars.(par);
        g_ub = new_rl_pars.(par) - rl.bounds.(par)(2);
        phi = phi + v * sum(max(0, g_lb) + max(0, g_ub));
        dphi = dphi + v * 0.5 * sum((sign(g_ub)) - sign(g_lb));
    end
end
