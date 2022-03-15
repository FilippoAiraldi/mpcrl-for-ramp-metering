function x_last_in = build_input(x0, x_last, M)
    % BUILD_INPUT Discard the first M values of x_last, repeats the last
    %   value M times, and put the initial value to x0.

    x_last_in = [x_last(:, M + 1:end), repmat(x_last(:, end), 1, M)];
    x_last_in(:, 1) = x0;
end