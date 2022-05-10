function V = Veq(rho, v_free, a, rho_crit, eps)
    % VEQ. Equilibrium speed in the METANET model
    arguments
        rho (:, :)
        v_free, a, rho_crit (1, 1) % double or symbolic
        eps (1, 1) double {mustBeNonnegative}
    end

    V = v_free * exp((-1 / a) * ((rho / rho_crit) + eps).^a);
end