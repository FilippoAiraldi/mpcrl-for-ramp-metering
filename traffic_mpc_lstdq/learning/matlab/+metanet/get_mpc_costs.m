function [TTS, Rate_var] = get_mpc_costs(n_links, n_origins, n_ramps, ...
                                                        Nc, T, L, lanes)
    % GET_MPC_COSTS. Returns the MPC's Total-Time-Spent cost, which
    % is widely used in traffic network control, and the cost associated
    % to the variability of the ramp metering rate control action.
    arguments
        n_links (1, 1) double {mustBePositive,mustBeInteger}
        n_origins (1, 1) double {mustBePositive,mustBeInteger}
        n_ramps (1, 1) double {mustBePositive,mustBeInteger}
        Nc (1, 1) double {mustBePositive,mustBeInteger}
        T, L (1, 1) double {mustBePositive}
        lanes (1, 1) double {mustBePositive,mustBeInteger}
    end

    % create symbolic arguments
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    r_last = casadi.SX.sym('r_last', n_ramps, 1);
    r = casadi.SX.sym('r', n_ramps, Nc);

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
