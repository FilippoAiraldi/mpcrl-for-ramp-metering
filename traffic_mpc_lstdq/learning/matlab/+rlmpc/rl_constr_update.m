function [pars, deltas] = rl_constr_update(pars, bounds, H, f, max_delta)
    if nargin < 5
        max_delta = 1 / 20;
    end

    % compute the bounds for the LCQP
    lb = struct;
    ub = struct;
    for name = fieldnames(pars)'
        max_d = abs(pars.(name{1}){end} * max_delta);
        lb.(name{1}) = max(...
            (bounds.(name{1})(1) - pars.(name{1}){end})', -max_d);
        ub.(name{1}) = min(...
            (bounds.(name{1})(2) - pars.(name{1}){end})', max_d);
    end
    lb = struct2array(lb);
    ub = struct2array(ub);

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
