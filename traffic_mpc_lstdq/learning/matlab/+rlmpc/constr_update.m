function [pars, deltas, lam_inf] = constr_update(pars, bnd, p, max_delta)
    % CONSTR_UPDATE. Performs the update of the learnable
    % parameters via a Linear Constrained Quadratic Programming problem,
    % ensuring that the parameter bounds are not compromised
    arguments
        pars (1, 1) struct
        bnd (1, 1) struct
        p (:, 1) double
        max_delta (1, 1) double = 1/10
    end

    % compute the bounds for the LCQP
    lb = struct;
    ub = struct;
    for name = fieldnames(pars)'
        rel_delta = abs(pars.(name{1}){end} * max_delta);
        rel_delta = max(rel_delta, 1e-1); % avoids parameters close to 0 not growing
        lb.(name{1}) = max(...
            bnd.(name{1})(:, 1) - pars.(name{1}){end}, -rel_delta)';
        ub.(name{1}) = min(...
            bnd.(name{1})(:, 2) - pars.(name{1}){end}, rel_delta)';
    end
    lb = struct2array(lb)';
    ub = struct2array(ub)';

    % solve constrained lcqp
    H = 0.5 * eye(length(p));
    [deltas, ~, exitflag, ~, lambda] = quadprog(H, -p, [], [], [], [], ...
        lb, ub, p, optimoptions('quadprog', 'Display', 'off', ...
                                    'Algorithm', 'interior-point-convex'));
    if exitflag == 0
        warning('quadprog exit flat 0')
    else
        assert(exitflag >= 1, 'quadprog failed (exit flag %i)', exitflag)
    end

    % compute next paramters
    i = 1;
    for name = fieldnames(pars)'
        par = name{1};
        sz = length(pars.(par){end});
        pars.(par){end + 1} = pars.(par){end} + deltas(i:i+sz-1);
        i = i + sz;
    end

    % compute infinity norm of multipliers
    lam_inf = norm([lambda.upper; lambda.lower], 'inf');
end
