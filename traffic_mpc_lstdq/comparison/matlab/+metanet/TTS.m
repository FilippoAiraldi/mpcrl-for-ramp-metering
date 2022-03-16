function J = TTS(w, rho, T, L, lanes)
    % TTS Total-Time Spent cost.
    
    J = T * sum(sum(w, 1) + sum(rho * L * lanes), 2);
end