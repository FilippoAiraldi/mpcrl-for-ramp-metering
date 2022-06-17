function [L, TTS, Rate_Var] = get_stage_cost(sim, model, mpc)
    % GET_STAGE_COST. Returns the Casadi function to compute the 
    % environment stage cost L (a.k.a., the opposite of the reward). Also 
    % returns the TTS and Rate Variability functions, which are part of the
    % stage cost.
    arguments
        sim (1, 1) struct
        model (1, 1) struct
        mpc (1, 1) struct
    end
    %
    n_origins = model.n_origins;
    n_links = model.n_links;
    n_ramps = model.n_ramps;
    L = model.L;
    max_queue = model.max_queue;
    lanes = model.lanes;
    %
    T = sim.T;
    %
    rate_var_penalty = mpc.rate_var_penalty;
    con_violation = mpc.con_violation_penalty;
    Nc = mpc.Nc;

    % create symbolic arguments
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    v = casadi.SX.sym('v', n_links, 1);
    r = casadi.SX.sym('r', n_ramps, Nc);
    r_prev = casadi.SX.sym('r_last', n_ramps, 1);

    % compute TTS
    J_TTS = T * (sum(w, 1) + sum(rho * L * lanes, 1));
    TTS = casadi.Function('tts_cost', ...
        {w, rho}, {J_TTS}, {'w', 'rho'}, {'TTS'});

    % compute Rate Variability
    J_RV = sum(sum(diff([r_prev, r], 1, 2).^2, 2), 1);
    Rate_Var = casadi.Function('rate_var_cost', ...
        {r_prev, r}, {J_RV}, {'r_last', 'r'}, {'rate_var'});

    % compute Stage Cost
    J_L = TTS(w, rho) ...
        + rate_var_penalty * Rate_Var(r_prev, r(:, 1)) ...
        + con_violation * max(0, w(2, :) - max_queue);
%         + con_violation^2 * ( ...
%               sum(max(0, -w), 1) ...
%             + sum(max(0, -rho), 1) ...
%             + sum(max(0, -v), 1)) ...
    L = casadi.Function('rl_cost', ...
        {w, rho, v, r(:, 1), r_prev}, {J_L}, ...
            {'w', 'rho', 'v', 'r', 'r_prev'}, {'L'});

    % make sure all are scalars
    assert(isequal(size(J_TTS), [1, 1]) && isequal(size(J_RV), [1, 1]) ...
           && isequal(size(J_L), [1, 1]));
end
