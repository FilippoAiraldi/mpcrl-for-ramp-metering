function V = Veq(rho, v_free, a, rho_crit)
    % VEQ. Equilibrium speed in the METANET model
    arguments
        rho (:, :) % double or symbolic
        v_free, a, rho_crit (1, 1) % double or symbolic
    end

    V = v_free * exp((-1 / a) * ((rho / rho_crit)).^a);
end
