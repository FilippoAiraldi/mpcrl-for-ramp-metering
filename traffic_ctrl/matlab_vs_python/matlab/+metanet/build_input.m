function x_last_in = build_input(x0, x_last, M)
    x_last_in = [x_last(:, M + 1:end), repmat(x_last(:, end), 1, M)];
    x_last_in(:, 1) = x0;
end