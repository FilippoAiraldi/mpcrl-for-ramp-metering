function Gm = modify_hessian(G)
    % MODIFY_HESSIAN Modifies the Hessian in a such a way to make it
    % positive definite, in case it is not.

%     [L, D] = util.mchol(G);
    [L, D] = util.modchol_ldlt(G);
    Gm = L * D * L';
end

