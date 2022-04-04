function L = get_rl_cost(n_origins, n_ramps, n_links, TTS, ...
        max_queue, con_violation_weight)
    
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);

    if n_ramps == 1
        L = TTS(w, rho) + ...
            con_violation_weight * max(0, w(2, :) - max_queue(2));
    else
        L = TTS(w, rho) + ...
            con_violation_weight * max(0, w(1, :) - max_queue(1)) + ...
            con_violation_weight * max(0, w(2, :) - max_queue(2));
    end
    assert(isequal(size(L), [1, 1]));

    L = casadi.Function('rl_cost', {w, rho}, {L}, {'w', 'rho'}, {'L'});
end


% function L = get_rl_cost(n_origins, n_links, TTS, ...
%         max_queue, con_violation_weight)
%     
%     w = casadi.SX.sym('w', n_origins, 1);
%     rho = casadi.SX.sym('rho', n_links, 1);
% 
%     L = TTS(w, rho) + con_violation_weight * max(0, w(2, :) - max_queue(2));
%     assert(isequal(size(L), [1, 1]));
% 
%     L = casadi.Function('rl_cost', {w, rho}, {L}, {'w', 'rho'}, {'L'});
% end
