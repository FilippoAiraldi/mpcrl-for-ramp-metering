function F = f2casadiF(T, L, lanes, C2, rho_max, tau, delta, eta, kappa, eps)
    % F2CASADIF Convert the metanet dynamics equation to a casadi function.

    % states, input and disturbances
    w = casadi.SX.sym('w', 2, 1);
    rho = casadi.SX.sym('rho', 3, 1);
    v = casadi.SX.sym('v', 3, 1);
    r = casadi.SX.sym('r', 1, 1);
    d = casadi.SX.sym('d', 3, 1);

    % parameters
    a = casadi.SX.sym('a', 1, 1);
    v_free = casadi.SX.sym('v_free', 1, 1);
    rho_crit = casadi.SX.sym('rho_crit', 1, 1);

    % run function
    [q_o, w_o_next, q, rho_next, v_next] = metanet.f(w, rho, v, [1; r], d, ...
        T, L, lanes, C2, rho_crit, rho_max, a, v_free, tau, delta, eta, ...
        kappa, eps);

    % create casadi function
    F = casadi.Function('F', {w, rho, v, r, d, a, v_free, rho_crit}, ...
        {q_o, w_o_next, q, rho_next, v_next}, ...
        {'w', 'rho', 'v', 'r', 'd', 'a', 'v_free', 'rho_crit'}, ...
        {'q_o', 'w_o_next', 'q', 'rho_next', 'v_next'});
end