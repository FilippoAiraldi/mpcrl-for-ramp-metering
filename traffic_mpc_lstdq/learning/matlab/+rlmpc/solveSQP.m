function sol = solveSQP(pars, vals, varnames, parnames, opts)
    % convert to vectors
    x0 = cellfun(@(n) vals.(n)(:), varnames, ...
                                    'UniformOutput', false);
    x0 = vertcat(x0{:});
    p = cellfun(@(n) pars.(n)(:), parnames, ...
                                    'UniformOutput', false);
    p = vertcat(p{:});

    % create functions
    obj = @(x) objective(p, x);
    nlcon = @(x) nonlcon(p, x);

    % solver
    [x, f, flag, output, lam_g] = fmincon(obj, ...
                            x0, [], [], [], [], [], [], nlcon, opts);

    % remove garbage from message
    output = split(output.message, '.');
    output = strtrim(output{1});

    % return struct
    sol = struct('x', x, 'p', p, 'f', f, 'lam_g', lam_g, ...
                                    'success', flag > 0, 'msg', output);
end



%% local functions
function [f, df] = objective(p, x)
    [f, df] = F_gen(p, x);
end

function [g_ineq, g_eq, dg_ineq, dg_eq] = nonlcon(p, x)
    [~, ~, g_eq, dg_eq, g_ineq, dg_ineq] = F_gen(p, x);
end




%% to include hessian
% function H = hessianfcn(x, lambda, mpc, p, H_sym)
%     lam_g = nan(size(mpc.opti.g));
%     lam_g(mpc.I_g.eq) = lambda.eqnonlin;
%     lam_g(mpc.I_g.ineq) = lambda.ineqnonlin;
% 
%     H = subsevalf(H_sym, ... 
%                 [mpc.opti.p; mpc.opti.x; mpc.opti.lam_g], [p; x; lam_g]);
% end

% H = hessian(mpc.opti.f + mpc.opti.lam_g' * mpc.opti.g, mpc.opti.x);

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
