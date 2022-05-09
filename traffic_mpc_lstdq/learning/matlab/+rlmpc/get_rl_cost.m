function L = get_rl_cost(n_links, n_origins, n_ramps, TTS, ...
        max_queue, con_violation_weight)
    % GET_RL_COST. Returns a casadi.Function for the RL stage cost
    % function.
    arguments
        n_links , n_origins, n_ramps ...
            (1, 1) double {mustBePositive,mustBeInteger}
        TTS (1, 1) casadi.Function
        max_queue (1, :) double
        con_violation_weight (1, 1) double {mustBePositive}
    end

    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);

    if n_ramps == 1
        L = TTS(w, rho) + ...
            con_violation_weight * max(0, w(2, :) - max_queue);
    else
        L = TTS(w, rho) + ...
            con_violation_weight * max(0, w(1, :) - max_queue(1)) + ...
            con_violation_weight * max(0, w(2, :) - max_queue(2));
    end
    assert(isequal(size(L), [1, 1]));

    L = casadi.Function('rl_cost', {w, rho}, {L}, {'w', 'rho'}, {'L'});
end
