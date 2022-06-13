function L = get_rl_cost(model, TTS, Rate_var, rate_var_penalty, con_violation)
    % GET_RL_COST. Returns a casadi.Function for the RL stage cost
    % function.
    arguments
        model (1, 1) struct
        TTS (1, 1) casadi.Function
        Rate_var (1, 1) casadi.Function
        rate_var_penalty (1, 1) double {mustBePositive}
        con_violation (1, 1) double {mustBePositive}
    end

    % create symbols
    w = casadi.SX.sym('w', model.n_origins, 1);
    rho = casadi.SX.sym('rho', model.n_links, 1);
    v = casadi.SX.sym('v', model.n_links, 1);
    r = casadi.SX.sym('r', model.n_ramps, 1);
    r_prev = casadi.SX.sym('r_prev', model.n_ramps, 1);
    
    % compute RL stage cost as instantaneous TTS + rate variability + 
    % constraint violation (both max queue and positivity - max w slacks is
    % punished less because the weight is learnable)
    L = TTS(w, rho) ...
        + rate_var_penalty * Rate_var(r_prev, r) ...
        + con_violation * max(0, w(2, :) - model.max_queue);
%         + con_violation^2 * ( ...
%               sum(max(0, -w), 1) ...
%             + sum(max(0, -rho), 1) ...
%             + sum(max(0, -v), 1)) ...
    assert(isequal(size(L), [1, 1]));
    
    % assemble as a function
    L = casadi.Function('rl_cost', ...
        {w, rho, v, r, r_prev}, {L}, ...
        {'w', 'rho', 'v', 'r', 'r_prev'}, {'L'});
end
