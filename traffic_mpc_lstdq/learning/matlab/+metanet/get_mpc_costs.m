function [TTS, Rate_var] = get_mpc_costs(model, mpc)
    % GET_MPC_COSTS. Returns the MPC's Total-Time-Spent cost, which
    % is widely used in traffic network control, and the cost associated
    % to the variability of the ramp metering rate control action.
    arguments
        model (1, 1) struct
        mpc (1, 1) struct
    end

    n_origins = model.n_origins;
    n_links = model.n_links;
    n_ramps = model.n_ramps;
    T = model.T;
    L = model.L;
    lanes = model.lanes;

    % create symbolic arguments
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    r_last = casadi.SX.sym('r_last', n_ramps, 1);
    r = casadi.SX.sym('r', n_ramps, mpc.pars.Nc);

    % compute TTS
    J_TTS = T * (sum(w, 1) + sum(rho * L * lanes, 1));
    assert(isequal(size(J_TTS), [1, 1]));
    TTS = casadi.Function('tts_cost', {w, rho}, {J_TTS}, ...
                                                    {'w', 'rho'}, {'TTS'});

    % compute rate variability
    J_RV = sum(sum(diff([r_last, r], 1, 2).^2, 2), 1);
    assert(isequal(size(J_RV), [1, 1]));
    Rate_var = casadi.Function('rate_var_cost', {r_last, r}, {J_RV}, ...
                                            {'r_last', 'r'}, {'rate_var'});
end
