function [p, H_mod] = descent_direction(g, H, version)
    % DESCENT_DIRECTION Computes the gradient descent direction from the
    % linear system 
    %                       H p = -g
    % where H is the hessian, p the descent direction and g the gradient. 
    arguments
        g (:, 1) double
        H (:, :) double = []
        version (1, 1) double ...
            {mustBeInteger, mustBeInRange(version, 0, 3)} = 0
    end
    assert(~isempty(H) || version == 0, 'hessian is required')
    switch version
        case 0
            % first order descent (no hessian information)
            p = -g;
            H_mod = nan; % no hessian modification
        case 1
            % cholesky with added multiple identities
            L = chol_multiple_identities(H, 1e-3);
            p = L' \ (L \ -g);  
            H_mod = norm(H - L * L', 'fro');
        case 2
            % modified cholesky factorization 1
            [L, D, E] = mchol(H);
            p = (D * L') \ (L \ -g); 
            H_mod = norm(E, 'fro');
        case 3
            % modified cholesky factorization 2
            [L, D] = modchol_ldlt(H);
            DL_ = D * L';
            p = DL_ \ (L \ -g); 
            H_mod = norm(H - L * DL_, 'fro');
        otherwise
            error('invalid hessian modification version')
    end
end


%% local functions
function L = chol_multiple_identities(G, beta)
    n = size(G, 1);
    g_min = min(diag(G));
    if g_min > 0
        tau = 0;
    else
        tau = -g_min + beta;
    end

    for i = 1:1e3
        try
            L = chol(G + tau * eye(n));
            return
        catch 
            tau = max(1.1 * tau, beta);
        end
    end
    error('Too many iterations')
end

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
end

function [L, DMC, P, D] = modchol_ldlt(A,delta)
    %modchol_ldlt  Modified Cholesky algorithm based on LDL' factorization.
    %   [L D,P,D0] = modchol_ldlt(A,delta) computes a modified
    %   Cholesky factorization P*(A + E)*P' = L*D*L', where 
    %   P is a permutation matrix, L is unit lower triangular,
    %   and D is block diagonal and positive definite with 1-by-1 and 2-by-2 
    %   diagonal blocks.  Thus A+E is symmetric positive definite, but E is
    %   not explicitly computed.  Also returned is a block diagonal D0 such
    %   that P*A*P' = L*D0*L'.  If A is sufficiently positive definite then 
    %   E = 0 and D = D0.  
    %   The algorithm sets the smallest eigenvalue of D to the tolerance
    %   delta, which defaults to sqrt(eps)*norm(A,'fro').
    %   The LDL' factorization is compute using a symmetric form of rook 
    %   pivoting proposed by Ashcraft, Grimes and Lewis.
    %   Reference:
    %   S. H. Cheng and N. J. Higham. A modified Cholesky algorithm based
    %   on a symmetric indefinite factorization. SIAM J. Matrix Anal. Appl.,
    %   19(4):1097-1110, 1998. doi:10.1137/S0895479896302898,
    %   Authors: Bobby Cheng and Nick Higham, 1996; revised 2015.
    if ~ishermitian(A), error('Must supply symmetric matrix.'), end
    if nargin < 2, delta = sqrt(eps)*norm(A,'fro'); end
    n = max(size(A));
    [L,D,p] = ldl(A,'vector'); 
    DMC = eye(n);
    % Modified Cholesky perturbations.
    k = 1;
    while k <= n
          if k == n || D(k,k+1) == 0 % 1-by-1 block
             if D(k,k) <= delta
                DMC(k,k) = delta;
             else
                DMC(k,k) = D(k,k);
             end
             k = k+1;
          
          else % 2-by-2 block
             E = D(k:k+1,k:k+1);
             [U,T] = eig(E);
             for ii = 1:2
                 if T(ii,ii) <= delta
                    T(ii,ii) = delta;
                 end
             end
             temp = U*T*U';
             DMC(k:k+1,k:k+1) = (temp + temp')/2;  % Ensure symmetric.
             k = k + 2;
          end
    end
    if nargout >= 3, P = eye(n); P = P(p,:); end
end
