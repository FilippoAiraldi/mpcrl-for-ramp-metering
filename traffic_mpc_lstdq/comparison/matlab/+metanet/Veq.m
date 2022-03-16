function V = Veq(rho, v_free, a, rho_crit)
    % VEQ Evaluates the METANET speed equation at the given density rho.
    
    V = v_free * exp((-1 / a) * (rho / rho_crit).^a);
end
