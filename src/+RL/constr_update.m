function [new_pars, deltas, lam_inf] = constr_update( ...
                                            pars, bnd, p, lr, max_delta)
    % CONSTR_UPDATE. Performs the update of the learnable
    % parameters via a Linear Constrained Quadratic Programming problem,
    % ensuring that the parameter bounds are not compromised
    arguments
        pars (1, 1) struct
        bnd (1, 1) struct
        p (:, 1) double
        lr (1, 1) double
        max_delta (1, 1) double
    end

    % compute the bounds for the LCQP
    parnames = string(fieldnames(pars)');
    lb = struct;
    ub = struct;
    for name = parnames
        rel_delta = abs(pars.(name) * max_delta);
        rel_delta = max(rel_delta, 1e-1); % avoids parameters close to 0 not growing
        lb.(name) = max(bnd.(name)(:, 1) - pars.(name), -rel_delta)';
        ub.(name) = min(bnd.(name)(:, 2) - pars.(name), rel_delta)';
        assert(all(lb.(name) <= ub.(name), 'all'))
    end
    lb = struct2array(lb)';
    ub = struct2array(ub)';

    % solve constrained lcqp
    H = eye(length(p));
    p = lr * p;
    [deltas, ~, exitflag, ~, lambda] = quadprog(H, -p, [], [], [], [], ...
        lb, ub, p, optimoptions('quadprog', 'Display', 'off', ...
                                    'Algorithm', 'interior-point-convex'));
    if exitflag == 0
        warning('quadprog exit flat 0')
    else
        assert(exitflag >= 1, 'quadprog failed (exit flag %i)', exitflag)
    end

    % compute next parameters
    i = 1;
    for name = parnames
        sz = length(pars.(name));
        new_pars.(name) = pars.(name) + deltas(i:i+sz-1);
        i = i + sz;
    end

    % compute infinity norm of multipliers
    if nargout > 2
        lam_inf = norm([lambda.upper; lambda.lower], 'inf');
    end
end