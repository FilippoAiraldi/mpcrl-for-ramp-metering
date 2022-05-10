function [f, p_opt] = get_Veq_approx(v_free, a, rho_crit, rho_max, eps)
    % GET_VEQ_APPROX. Returns a casadi.Function that works as a 2-piece
    % piecewise affine approximation of the nonlinear equilibrium speed
    % equation, Veq(rho).
    arguments
        v_free, a, rho_crit, rho_max (1, 1) double {mustBeNonnegative}
        eps (1, 1) double {mustBeNonnegative} = 0
    end

    % build the symbolic appproximation of Veq
    f = get_Veq_approx_internal();

    % compute its optimal parameters
    % build the real curve
    x = linspace(0, rho_max, 1e2);
    y = metanet.Veq(x, v_free, a, rho_crit, eps);

    % approximation via Matlab
    [p_opt, ~, ~, flag] = lsqcurvefit(...
        @(p, xi) full(f(xi, p)), ...                        % curve to fit in p
        [-1, v_free, 1], ...                                % inital guess
        x, y, ...                                           % data coordinates
        [-1e2, 1e-2, 5], [1e-2, v_free * 2, v_free], ...    % lb and ub for p
        optimoptions('lsqcurvefit', 'Display', 'none'));                  
    assert(flag > 0, 'Veq approximation failed (flag=%i)', flag)
    if isrow(p_opt)
        p_opt = p_opt';
    end

    % plot comparison
    % plot(x, y, x, full(f(x, p_opt)))
end


function f = get_Veq_approx_internal()
    % input
    rho = casadi.SX.sym('rho', 1, 1);

    % parameters
    % v_free = casadi.SX.sym('v_free', 1, 1);
    pars = casadi.SX.sym('pars_Veq_approx', 3, 1);

    % output
    v = max(pars(1) * rho + pars(2), pars(3));
    % v = max(min(v_free, pars(1) * rho + pars(2)), pars(3));

    % function
    f = casadi.Function('Veq_approx', ...
        {rho, pars}, {v}, ...
        {'rho', 'pars_Veq_approx'}, {'v'});
end
