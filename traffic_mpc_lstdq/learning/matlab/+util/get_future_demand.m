function d = get_future_demand(D, ep, k, K, M, Np) 
%     if k <= K - M * Np + 1
%         d = D(:, k:k + M * Np - 1);
%     else
%         d = [D(:, k:end), repmat(D(:, end), 1, M * Np - 1 - K + k)];
%     end
    idx = k + (ep - 1) * K;
    d = D(:, idx:idx + M * Np - 1);
end

