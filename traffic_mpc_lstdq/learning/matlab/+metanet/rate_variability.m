function J = rate_variability(n_ramps, Nc)
    % RATE_VARIABILITY. Computes a casadi.Function for the cost associated
    % to the variability of the ramp metering rate control action.
    arguments
        n_ramps (1, 1) double {mustBePositive,mustBeInteger}
        Nc (1, 1) double {mustBePositive,mustBeInteger}
    end

    r_last = casadi.SX.sym('r_last', n_ramps, 1);
    r = casadi.SX.sym('r', n_ramps, Nc);

    variability = sum(sum(diff([r_last, r], 1, 2).^2, 2), 1);
    assert(isequal(size(variability), [1, 1]));

    J = casadi.Function('rate_var_cost', {r_last, r}, {variability}, ...
        {'r_last', 'r'}, {'rate_var'});
end