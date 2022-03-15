function J = input_variability_penalty(r_last, r)
   J = sum(diff([r_last, r], 1, 2).^2, 2);
end