 function [sol, get_value, x_opt, fval, p, g, lam_g, flag, output] = ...
                                                solveSQP(mpc, vals)
    arguments
        mpc (1, 1) rlmpc.NMPC
        vals (1, 1) struct
    end

    % pre-compute stuff
    x0 = subsevalf(mpc.opti.x, mpc.vars, vals);
    p = mpc.opti.value(mpc.opti.p);
    df = jacobian(mpc.opti.f, mpc.opti.x)';
%     H = hessian(mpc.opti.f + mpc.opti.lam_g' * mpc.opti.g, mpc.opti.x);
    g = mpc.opti.g;
    g_eq = g(mpc.I_g.eq);
    dg_eq = jacobian(g_eq, mpc.opti.x)';
    g_ineq = g(mpc.I_g.ineq);
    dg_ineq = jacobian(g_ineq, mpc.opti.x)';

    % objective 
    obj = @(x) objective(x, mpc, p, df);

    % non linear constraints - dynamics
    nlcon = @(x) nonlcon(x, mpc, p, g_eq, dg_eq, g_ineq, dg_ineq);

    % solve the sqp problem
    opts = optimoptions('fmincon', 'Algorithm', 'sqp', ...
                        'Display', 'none', ...
                        'OptimalityTolerance', 1e-8, ...
                        'StepTolerance', 1e-8, ...
                        'ScaleProblem', true, ...
                        'SpecifyObjectiveGradient', true, ...
                        'SpecifyConstraintGradient', true);
    [x_opt, fval, flag, output, lam_g_struct] = fmincon(obj, ...
                            x0, [], [], [], [], [], [], nlcon, opts);
%     if flag <= 0
%         % fall back to interior point
%         opts.Algorithm = 'interior-point';
%         opts.HessianFcn = @(x, l) hessianfcn(x, l, mpc, p, H);
%         opts.BarrierParamUpdate = 'predictor-corrector';
%         opts.EnableFeasibilityMode = true;
%         opts.InitBarrierParam = 1e-2;
%         opts.MaxIterations = 400;
%         [x_opt2, fval, flag, output, lam_g_struct] = fmincon(obj, ...
%                             x_opt, [], [], [], [], [], [], nlcon, opts);
%         assert(flag > 0, 'AAAAAAHHHHH')
%     end

    % put multiplier in a unique vector
    lam_g = nan(size(mpc.opti.g));
    lam_g(mpc.I_g.eq) = lam_g_struct.eqnonlin;
    lam_g(mpc.I_g.ineq) = lam_g_struct.ineqnonlin;

    % build a function to get the values 
    get_value = @(o) subsevalf(o, ...
            [mpc.opti.p; mpc.opti.x; mpc.opti.lam_g], [p; x_opt; lam_g]);

    % return constraint value at the optimal
    g = get_value(mpc.opti.g);

    % compute per-variable solution
    sol = struct;
    for n = fieldnames(vals)'
        sol.(n{1}) = get_value(mpc.vars.(n{1}));
    end
end



%%
function y = subsevalf(expr, old, new)
    y = expr;

    if isstruct(old)
        assert(isstruct(new))
        for n = fieldnames(old)'
            y = casadi.substitute(y, old.(n{1}), new.(n{1}));
        end

    elseif iscell(old)
        assert(iscell(new))
        for n = 1:length(cell)
            y = casadi.substitute(y, old{n}, new{n});
        end
    else
        y = casadi.substitute(y, old, new);
    end

    y = full(evalf(y));
end

function [f, df] = objective(x, mpc, p, df_sym)
    % compute objective
    f = subsevalf(mpc.opti.f, [mpc.opti.p; mpc.opti.x], [p; x]);

    % compute objective gradient
    if nargout == 1
        return
    end
    df = subsevalf(df_sym, [mpc.opti.p; mpc.opti.x], [p; x]);
end

function [c, ceq, dc, dceq] = nonlcon(x, mpc, p, ...
                                            g_eq, dg_eq, g_ineq, dg_ineq)
    % nonlinear inequalities
    c = subsevalf(g_ineq, [mpc.opti.p; mpc.opti.x], [p; x]);

    % nonlinear equalities 
    ceq = subsevalf(g_eq, [mpc.opti.p; mpc.opti.x], [p; x]);

    % compute the constraint gradients
    if nargout == 2
        return
    end
    dc = subsevalf(dg_ineq, [mpc.opti.p; mpc.opti.x], [p; x]);
    dceq = subsevalf(dg_eq, [mpc.opti.p; mpc.opti.x], [p; x]);
end

function H = hessianfcn(x, lambda, mpc, p, H_sym)
    lam_g = nan(size(mpc.opti.g));
    lam_g(mpc.I_g.eq) = lambda.eqnonlin;
    lam_g(mpc.I_g.ineq) = lambda.ineqnonlin;

    H = subsevalf(H_sym, [mpc.opti.p; mpc.opti.x; mpc.opti.lam_g], ...
                                                            [p; x; lam_g]);
end



%%
%  function [x_opt, fval, flag, output, lam_g] = solveSQP(mpc, pars, vals)
%     arguments
%         mpc (1, 1) rlmpc.NMPC
%         pars (1, 1) struct
%         vals (1, 1) struct
%     end
% 
%     % build init.vals, A, b, lb, ub, Aeq and beq 
%     varnames = fieldnames(mpc.vars);
%     args = cell(length(varnames), 5);
%     for i = 1:size(args, 1)
%         var = varnames{i};
%         sz = size(mpc.vars.(var));
%         n_el = prod(sz);
%         
%         % init.vals for solver
%         args{i, 1} = vals.(var)(:); 
%         
%         % lb, ub
%         switch var
%             case 'r'
%                 args(i, 2:3) = {ones(n_el, 1) * 0.2, ones(n_el, 1) * 1};
%             case 'slack'
%                 args(i, 2:3) = {ones(n_el, 1) * eps^2, inf(n_el, 1)};
%             otherwise
%                 args(i, 2:3) = {ones(n_el, 1) * eps, inf(n_el, 1)};
%         end
% 
%         % Aeq, beq - MPC initial conditions x0
%         if isfield(pars, [var, '0'])
%             Aeq = [eye(sz(1)), zeros(sz(1), n_el - sz(1))];
%             beq = pars.([var, '0']);
%             args(i, 4:5) = {Aeq, beq};
%         end
%     end
%     init_vals = vertcat(args{:, 1}); % initial conditions for solvers
%     lb = vertcat(args{:, 2});
%     ub = vertcat(args{:, 3});
%     Aeq = blkdiag(args{:, 4}); 
%     Aeq = [Aeq, zeros(size(Aeq, 1), mpc.opti.nx - size(Aeq, 2))];
%     beq = vertcat(args{:, 5});
% 
%     % A, b - soft max queue constraints
%     A = repmat({diag(isfinite(mpc.max_queue))}, 1, size(vals.w, 2)); 
%     A = blkdiag(A{:});
%     A = double(A(any(A  ~= 0, 2), :)); % remove all zeros rows
%     A = [A, ...
%         zeros(size(A, 1), length(lb) - size(A, 2) - length(vals.slack(:))), ...
%         -eye(size(A, 1))]; 
%     b = repmat(mpc.max_queue(:), size(vals.w, 2), 1);
%     b = b(isfinite(b));
% 
%     % convert parameters in a single vector
%     p = mpc.opti.value(mpc.opti.p);
% 
%     % objective 
%     obj = @(x) objective(x, mpc, p);
% 
%     % non linear constraints - dynamics
%     nlcon = @(x) nonlcon(x, mpc, p);
%     
%     % solve the sqp problem
%     [x_opt, fval, flag, output, lam_g] = fmincon(obj, ...
%         init_vals, A, b, Aeq, beq, lb, ub, nlcon, ...
%         optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'final', ...
%                      'ScaleProblem', true, ...
%                      'SpecifyObjectiveGradient', true, ...
%                      'SpecifyConstraintGradient', true));
%     % SpecifyConstraintGradient, SpecifyObjectiveGradient
%     assert(flag > 0, 'AAAHHHHH');
% end
% 
% 
% 
% %%
% function [f, df] = objective(x, mpc, p)
%     % compute objective
%     f = casadi.substitute(mpc.opti.f, [mpc.opti.p; mpc.opti.x], [p; x]);
%     f = full(evalf(f));
% 
%     % compute objective gradient
%     if nargout == 1
%         return
%     end
%     df = jacobian(mpc.opti.f, mpc.opti.x)';
%     df = casadi.substitute(df, [mpc.opti.p; mpc.opti.x], [p; x]);
%     df = full(evalf(df));
% end
% 
% function [c, ceq, dc, dceq] = nonlcon(x, mpc, p)
%     % no nonlinear inequalities
%     c = [];      % c(x) <= 0
%     dc = [];
% 
%     % nonlinear equalities due to dynamics (from 265 to 456)
%     g = mpc.opti.g;
%     g = g(265:456);
%     ceq = casadi.substitute(g, [mpc.opti.p; mpc.opti.x], [p; x]);
%     ceq = full(evalf(ceq));
% 
%     % compute the constraint gradient
%     if nargout == 2
%         return
%     end
%     dceq = jacobian(g, mpc.opti.x)';
%     dceq = casadi.substitute(dceq, [mpc.opti.p; mpc.opti.x], [p; x]);
%     dceq = full(evalf(dceq));
% end
