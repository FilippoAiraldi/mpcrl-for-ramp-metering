function [pars, deltas] = rl_constrained_update(pars, bnd, H, f, max_delta)
    % RL_CONSTRAINED_UPDATE. Performs the update of the learnable
    % parameters via a Linear Constrained Quadratic Programming problem,
    % ensuring that the parameter bounds are not compromised
    arguments
        pars (1, 1) struct
        bnd (1, 1) struct
        H (:, :) double
        f (:, 1) double
        max_delta (1, 1) double = 1/10
    end

    % compute the bounds for the LCQP
    lb = struct;
    ub = struct;
    for name = fieldnames(pars)'
        rel_delta = abs(pars.(name{1}){end} * max_delta);
        rel_delta = max(rel_delta, 1e-1);
        lb.(name{1}) = max(...
            bnd.(name{1})(1) - pars.(name{1}){end}, -rel_delta)';
        ub.(name{1}) = min(...
            bnd.(name{1})(2) - pars.(name{1}){end}, rel_delta)';
    end
    lb = struct2array(lb)';
    ub = struct2array(ub)';

    % solve constrained lcqp
    [deltas, ~, exitflag] = quadprog(H, f, [], [], [], [], lb, ub, -f, ...
        optimoptions('quadprog', 'Display', 'off', ...
            'Algorithm', 'active-set'));
    assert(exitflag >= 1, 'quadprog failed (exit flag %i)', exitflag)

    % compute next paramters
    i = 1;
    for name = fieldnames(pars)'
        par = name{1};
        sz = length(pars.(par){end});
        pars.(par){end + 1} = pars.(par){end} + deltas(i:i+sz-1);
        i = i + sz;
    end
end
