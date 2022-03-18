function J = TTS(w, rho, T, L, lanes)
    J = T * (sum(w, 1) + sum(rho * L * lanes, 1));
end