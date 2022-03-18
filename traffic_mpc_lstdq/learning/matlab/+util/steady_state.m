function [w_ss, rho_ss, v_ss] = steady_state(F, w0, rho0, v0, r, d, a, v_free, rho_crit, tol, maxiter)
    % STEADY_STATE Runs the given dynamic function until convergence to a
    %   steady-state is detected.
    
    if nargin < 10
        tol = 1e-3;
    end
    if nargin < 11
        maxiter = 2e3;
    end
    i = 0;
    err = inf;

    while i < maxiter && err >= tol
        % step the system
        [~, w_ss, ~, rho_ss, v_ss] = F(...
            w0, rho0, v0, r, d, a, v_free, rho_crit);
        w_ss = full(w_ss);
        rho_ss = full(rho_ss);
        v_ss = full(v_ss);


        % compute convergence error
        err = sum(abs(w_ss - w0)) + ...
            sum(abs(rho_ss - rho0)) + sum(abs(v_ss - v0));
        w0 = w_ss;
        rho0 = rho_ss;
        v0 = v_ss;
        i = i + 1;
    end
end
