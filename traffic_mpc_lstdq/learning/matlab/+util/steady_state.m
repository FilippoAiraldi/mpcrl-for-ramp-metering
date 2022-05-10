function [w_ss, rho_ss, v_ss, err, k] = steady_state( ...
                                                F, w0, rho0, v0, r, d, ...
                                                rho_crit, a, v_free, ...
                                                tol, maxiter)
    % STEADY_STATE. Runs the given dynamic function until convergence to a
    % steady-state is detected.
    arguments
        F (1, 1) casadi.Function
        w0, rho0, v0, r, d (:, 1) double
        rho_crit, a, v_free (1, 1) double {mustBeNonnegative}
        tol (1, 1) double {mustBeNonnegative} = 1e-8
        maxiter (1, 1) double {mustBeNonnegative,mustBeInteger} = 1e3
    end

    k = 0;
    err = inf;
    while k < maxiter && err >= tol
        % step the system
        [~, w_ss, ~, rho_ss, v_ss] = F(w0, rho0, v0, r, d, ...
                                                    rho_crit, a, v_free);
        w_ss = full(w_ss);
        rho_ss = full(rho_ss);
        v_ss = full(v_ss);

        % compute convergence error
        err = norm(w_ss - w0) + norm(rho_ss - rho0) + norm(v_ss - v0);
        w0 = w_ss;
        rho0 = rho_ss;
        v0 = v_ss;
        k = k + 1;
    end
end
