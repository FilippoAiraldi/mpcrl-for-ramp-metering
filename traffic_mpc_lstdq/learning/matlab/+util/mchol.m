function [L, D, E] = mchol(G)
    % [L, D, E] = MCHOL(G). Modified Cholesky factorization. Given a 
    % symmetric matrix G, possibly non positive definite, find a matrices E 
    % (of small norm), L, and D such that G + E is positive definite and 
    %
    %      G + E = L*D*L'
    %
    % Reference: Gill, Murray, and Wright, "Practical Optimization", 
    % page 111.
    % 
    % Inspired by https://nl.mathworks.com/matlabcentral/fileexchange/47-ldlt


    % MC1
    n = size(G, 1);
    gamma = max(diag(G));
    xi = max(G(not(eye(n))));
    beta2 = max([gamma, xi / max(1, sqrt(n^2 - 1)), 1e-12]);

    % MC2
    L = zeros(n);
    D = zeros(n);
    E = zeros(n);
    theta = zeros(n, 1);
    C = diag(diag(G));
    for j = 1:n
        bb = 1:j - 1;
        ee = j + 1:n;
    
        % MC4
        if j > 1
            L(j, bb) = C(j, bb) ./ diag(D(bb, bb))';
        end
        if j >= 2
            if j ~= n
                C(ee, j) = G(ee, j) - (L(j, bb) * C(ee, bb)')';
            end
        else
            C(ee, j) = G(ee, j);
        end
        if j == n
            theta(j) = 0;
        else
            theta(j) = max(abs(C(ee, j)));
        end

        % MC5
        D(j, j) = max([eps, abs(C(j, j)), theta(j)^2 / beta2]);
        E(j, j) = D(j, j) - C(j, j);
        
        % MC6       
        ind=(j * (n + 1) + 1:n + 1:n^2)';
        C(ind) = C(ind) - C(ee, j).^2 / D(j, j);
    end
    ind = (1:n + 1:n^2)';
    L(ind) = 1;
return
