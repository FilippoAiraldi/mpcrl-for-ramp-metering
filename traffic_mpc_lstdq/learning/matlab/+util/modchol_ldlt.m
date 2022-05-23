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
