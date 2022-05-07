function [pars, deltas] = rl_constr_update(pars, bounds, f)
    % compute the bounds for the LCQP
    lb = struct;
    ub = struct;
    for name = fieldnames(pars)'
        lb.(name{1}) = (bounds.(name{1})(1) - pars.(name{1}){end})';
        ub.(name{1}) = (bounds.(name{1})(2) - pars.(name{1}){end})';
    end
    lb = struct2array(lb);
    ub = struct2array(ub);

    % solve constrained lcqp
    H = eye(length(f));
    [deltas, ~, exitflag] = quadprog(H, f, [], [], [], [], lb, ub, -f, ...
        optimoptions('quadprog', 'Display', 'off', ...
            'Algorithm', 'active-set'));
    assert(exitflag >= 1)

    % compute next paramters
    i = 1;
    for name = fieldnames(pars)'
        par = name{1};
        sz = length(pars.(par){end});
        pars.(par){end + 1} = pars.(par){end} + deltas(i:i+sz-1);
        i = i + sz;
    end
end
