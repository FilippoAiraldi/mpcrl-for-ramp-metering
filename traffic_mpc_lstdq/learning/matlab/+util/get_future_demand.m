function d = get_future_demand(D, k, K, M, Np)
    % d = GET_FUTURE_DEMAND(D, k, K, M, Np) From the total demands D, gets
    %   the demand from k to k+M*Np+1. Pads the result if k is close 
    %   to the length of D.

    if k <= K - M * Np + 1
        d = D(:, k:k + M * Np - 1);
    else
        d = [D(:, k:end), repmat(D(:, end), 1, M * Np - 1 - K + k)];
    end
end

