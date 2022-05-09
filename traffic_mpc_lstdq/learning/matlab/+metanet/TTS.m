function J = TTS(n_links, n_origins, T, L, lanes)
    % TTS. Computes a casadi.Function for the Total-Time-Spent cost, which
    % is widely used in traffic network control.
    arguments
        n_links (1, 1) double {mustBePositive,mustBeInteger}
        n_origins (1, 1) double {mustBePositive,mustBeInteger}
        T, L (1, 1) double {mustBePositive}
        lanes (1, 1) double {mustBePositive,mustBeInteger}
    end

    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);

    TTS = T * (sum(w, 1) + sum(rho * L * lanes, 1));
    assert(isequal(size(TTS), [1, 1]));
    
    J = casadi.Function('tts_cost', {w, rho}, {TTS}, ...
        {'w', 'rho'}, {'TTS'});
end