function J = rate_variability(r_last, r)
   J = sum(diff([r_last, r], 1, 2).^2, 2);
end