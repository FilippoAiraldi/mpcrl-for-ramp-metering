function J = input_variability_penalty(r_last, r)
    % INPUT_VARIABILITY_PENALTY Cost function term due to input variability
    %   in the ramp metering rate.
    
   J = sum(diff([r_last, r], 1, 2).^2, 2);
end