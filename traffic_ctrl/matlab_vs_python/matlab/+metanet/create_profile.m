function profile = create_profile(t, x, y)
    % creates a profile passing through points (x, y) along time t
    arguments
        t (1, :) double
        x (1, :) double
        y (1, :) double
    end

    [~, x] = min(abs(t - x'), [], 2);
    if x(1) ~= 1
        x = [1, x', length(t)];
        y = [y(1), y, y(end)];
    else
        x = [x', length(t)];
        y = [y, y(end)];        
    end

    profile = zeros(size(t));
    for i = 1:length(x) - 1
        m = (y(i + 1) - y(i)) / (t(x(i + 1)) - t(x(i)));
        q = y(i) - m * t(x(i));
        profile(x(i):x(i + 1)) = m * t(x(i):x(i + 1)) + q;
    end
    profile(end) = profile(end - 1);
end