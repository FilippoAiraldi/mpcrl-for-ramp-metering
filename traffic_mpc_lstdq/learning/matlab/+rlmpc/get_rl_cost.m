function L = get_rl_cost(model, mpc, TTS, Rate_var)
    % GET_RL_COST. Returns a casadi.Function for the RL stage cost
    % function.
    arguments
        model (1, 1) struct
        mpc (1, 1) struct
        TTS (1, 1) casadi.Function
        Rate_var (1, 1) casadi.Function
    end
    n_links = model.n_links;
    n_origins = model.n_origins;
    n_ramps = model.n_ramps;
    max_queue = model.max_queue;
    rate_var_penalty = mpc.rate_var_penalty;
    con_violation = mpc.con_violation_penalty;

    % create symbols
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    v = casadi.SX.sym('v', n_links, 1);
    r = casadi.SX.sym('r', n_ramps, 1);
    r_prev = casadi.SX.sym('r_prev', n_ramps, 1);
    
    % compute RL stage cost as instantaneous TTS + rate variability + 
    % constraint violation (both max queue and positivity - max w slacks is
    % punished less because the weight is learnable)
    L = TTS(w, rho) ...
        + rate_var_penalty * Rate_var(r_prev, r) ...
        + con_violation * max(0, w(2, :) - max_queue);
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
