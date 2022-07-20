function [L, TTS, RV] = get_stage_cost(sim, model, mpc)
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
    RV_penalty = mpc.RV_penalty;
    CV_penalty = mpc.CV_penalty;
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
    J_RV = sum(sum(diff([r_prev, r] / mpc.norm.r, 1, 2).^2, 2), 1);
    RV = casadi.Function('rate_var_cost', ...
        {r_prev, r}, {J_RV}, {'r_last', 'r'}, {'RV'});

    % compute Stage Cost
    J_TTS = TTS(w, rho);                        % already computed; just to be on the safe side
    J_RV = RV_penalty * RV(r_prev, r(:, 1));    % just on current r
    J_con = CV_penalty * max(0, w(2, :) - max_queue);
    %         + CV_penalty * ( ...
    %               sum(max(0, -w), 1) ...
    %             + sum(max(0, -rho), 1) ...
    %             + sum(max(0, -v), 1)) ...
    J_L = J_TTS + J_RV + J_con;
    L = casadi.Function('stage_cost', ...
        {w, rho, v, r(:, 1), r_prev}, {J_L, J_TTS, J_RV, J_con}, ...
        {'w', 'rho', 'v', 'r', 'r_prev'}, {'L', 'TTS', 'RV', 'ConViol'});

    % make sure all are scalars
    assert(isequal(size(J_TTS), [1, 1]) && isequal(size(J_RV), [1, 1]) ...
           && isequal(size(J_L), [1, 1]));
end
