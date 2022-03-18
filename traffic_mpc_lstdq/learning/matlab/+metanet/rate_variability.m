function J = rate_variability(r_last, r)
    % RATE_VARIABILITY Cost function term due to input variability
    %   in the ramp metering rate.
    
   J = sum(diff([r_last, r], 1, 2).^2, 2);
end