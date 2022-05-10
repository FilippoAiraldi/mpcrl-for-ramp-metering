function L = get_rl_cost(n_links, n_origins, TTS, ...
                                                max_queue, con_violation)
    % GET_RL_COST. Returns a casadi.Function for the RL stage cost
    % function.
    arguments
        n_links , n_origins ...
            (1, 1) double {mustBePositive,mustBeInteger}
        TTS (1, 1) casadi.Function
        max_queue (:, 1) double
        con_violation (1, 1) double {mustBePositive}
    end
    assert(length(max_queue) == n_origins)

    % create symbols
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    
    % compute cost
    L = TTS(w, rho);
    for i = 1:n_origins
        if isfinite(max_queue(i))
            L = L + con_violation * max(0, w(i, :) - max_queue(i));
        end
    end
    assert(isequal(size(L), [1, 1]));
    L = casadi.Function('rl_cost', {w, rho}, {L}, {'w', 'rho'}, {'L'});
end
