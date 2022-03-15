function J = TTS(w, rho, T, L, lanes)
    J = T * sum(sum(w, 1) + sum(rho * L * lanes), 2);
end