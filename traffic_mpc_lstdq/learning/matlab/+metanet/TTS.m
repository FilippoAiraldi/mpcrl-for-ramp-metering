function J = TTS(n_origins, n_links, T, L, lanes)

    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);

    TTS = T * (sum(w, 1) + sum(rho * L * lanes, 1));
    assert(isequal(size(TTS), [1, 1]));
    
    J = casadi.Function('tts_cost', {w, rho}, {TTS}, ...
        {'w', 'rho'}, {'J_TTS'});
end