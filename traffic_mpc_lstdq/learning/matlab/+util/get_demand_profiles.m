function D = get_demand_profiles(time, episodes, type)
    % GET_DEMAND_PROFILES Returns the demand profiles for the network, 
    % either fixed or random.
    arguments
        time (1, :) double
        episodes (1, 1) {mustBeInteger, mustBePositive}
        type {mustBeMember(type, {'fixed', 'random'})} = 'random'
    end

    if strcmp(type, 'fixed')
        d1 = create_profile(time, [0, .35, 1, 1.35], [1e3, 3e3, 3e3, 1e3]);
        d2 = create_profile(time, [.15,.35,.6,.8], [500, 1500, 1500, 500]);
        d_cong = create_profile(time, [0.5, .7, 1, 1.2], [20, 60, 60, 20]);
        D = repmat([d1; d2; d_cong], 1, episodes);
    else
        d1_h = rand_between([1, episodes], 2000, 3500);
        d1_c = rand_between([1, episodes], 0.5, 0.9);
        d1_w1  = rand_between([1, episodes], 0.2, 0.4) / 2;
        d1_w2  = rand_between(size(d1_w1), d1_w1 * 1.5, d1_w1 * 3);
        d2_h = rand_between([1, episodes], 1000, 2000);
        d2_c = rand_between([1, episodes], 0.3, 0.7);
        d2_w1  = rand_between([1, episodes], 0.1, 0.4) / 2;
        d2_w2  = rand_between(size(d2_w1), d2_w1 * 1.5, d2_w1 * 3);
        d3_h = rand_between([1, episodes], 40, 80);
        d3_c = rand_between([1, episodes], 0.7, 1);
        d3_w1  = rand_between([1, episodes], 0.1, 0.5) / 2;
        d3_w2  = rand_between(size(d3_w1), d3_w1 * 1.5, d3_w1 * 3);
        
        D = cell(3, episodes);
        for i = 1:episodes
            D{1, i} = create_profile(time, ...
                [d1_c(i) - d1_w2(i), d1_c(i) - d1_w1(i), ...
                            d1_c(i) + d1_w1(i), d1_c(i) + d1_w2(i)], ...
                [1000, d1_h(i), d1_h(i), 1000]);
            D{2, i} = create_profile(time, ...
                [d2_c(i) - d2_w2(i), d2_c(i) - d2_w1(i), ...
                            d2_c(i) + d2_w1(i), d2_c(i) + d2_w2(i)], ...
                [500, d2_h(i), d2_h(i), 500]);
            D{3, i} = create_profile(time, ...
                [d3_c(i) - d3_w2(i), d3_c(i) - d3_w1(i), ...
                            d3_c(i) + d3_w1(i), d3_c(i) + d3_w2(i)], ...
                [20, d3_h(i), d3_h(i), 20]);
        end
        D = cell2mat(D);
    end
    
    % add noise and then filter to make it more realistic
    [filter_num, filter_den] = butter(3, 0.1);
    D = filtfilt(filter_num, filter_den, ...
                                (D + randn(size(D)) .* [95; 95; 1.7])')';
end



%% local function
function profile = create_profile(t, x, y)
    % CREATE_PROFILE. Creates a profile passing through points (x, y) along 
    % time t.
    arguments
        t (1, :) double
        x (1, :) double
        y (1, :) double
    end
    assert(isequal(size(x), size(y)), 'x and y do not share the same size')

    x(1) = max(x(1), t(1));
    x(end) = min(x(end), t(end));
    assert(issorted(x), 'x is ill-specified')
    % assert(x(1) >= t(1) && x(end) <= t(end), 'x must contained within t')

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

function r = rand_between(size, a, b)
    r = rand(size) .* (b - a) + a;
end
