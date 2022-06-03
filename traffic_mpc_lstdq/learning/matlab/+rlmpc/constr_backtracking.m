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
        k_worst (1, 1) double = 10
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
    c1 = 0.25;
    c2 = 0.9;
    rho = 0.2;
    [phi0, dphi0] = eval_phi(0);
    dphi0 = p' * dphi0;
    lr = 1;
    for i = 1:nsteps
        try
            [phi, dphi] = eval_phi(lr);
            if (phi <= phi0 + c1 * lr * dphi0) && (dphi' * p >= c2 * dphi0)
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
function [phi, dphi]  = evaluate_phi(alpha, p, target, Q, derivQ, pars, ...
                                                      vals, rl, v)
    N = length(target);
    f = zeros(N, 1);
    dQ = zeros(size(derivQ.dL, 1), N);
    parnames = string(fieldnames(rl.pars)');
    
    % modify the RL parameters by alpha
    new_rl_pars = struct;
    for par = parnames
        new_par = rl.pars.(par){end} + alpha * p.(par);
%         if ~strcmp(par, 'weight_V')
%             new_par = max(1e-3, new_par); % all parameters expect weight_V must be positive
%         end
        new_rl_pars.(par) = new_par;
    end
    all_failed = true;
    for i = 1:N
        for par = parnames
            pars(i).(par) = new_rl_pars.(par);
        end

        % solve the MPC problem - don't shift values as they are already the
        % solution, and don't use multistart
        [~, ~, info] = evalc('Q.solve(pars(i), vals(i), false, 1)');
        
        % compute some values
        if info.success
            f(i) = info.f;
            dQ(:, i) = info.get_value(derivQ.dL);
            all_failed = false;
        end
    end
    assert(~all_failed, 'Evaluation of phi failed')

    % compute the value of phi and its derivative
    phi = sum((target - f).^2);
    dphi = - dQ * (target - f);
    for name = fieldnames(rl.pars)'
        par = name{1};

        % compute the values of the boundary constraints
        g_lb = rl.bounds.(par)(1) - new_rl_pars.(par);
        g_ub = new_rl_pars.(par) - rl.bounds.(par)(2);

        % include boundaries into the objective
        phi = phi + v * sum(max(0, g_lb) + max(0, g_ub));
        dphi = dphi + v * 0.5 * sum((sign(g_ub)) - sign(g_lb));
    end
end
