function F = f2casadiF(config)
    w = casadi.SX.sym('w', 2, 1);
    rho = casadi.SX.sym('rho', 3, 1);
    v = casadi.SX.sym('v', 3, 1);
    r2 = casadi.SX.sym('r', 1, 1);
    d = casadi.SX.sym('d', 2, 1);

    [q_o, w_o_next, q, rho_next, v_next] = metanet.f(w, rho, v, r2, d, config);

    F = casadi.Function('F', {w, rho, v, r2, d}, ...
        {q_o, w_o_next, q, rho_next, v_next}, ...
        {'w', 'rho', 'v', 'r', 'd'}, ...
        {'q_o', 'w_o_next', 'q', 'rho_next', 'v_next'});
end