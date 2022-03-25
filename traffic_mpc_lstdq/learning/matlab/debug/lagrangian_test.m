clc, close all, clear all

% init
opti = casadi.Opti();
x = opti.variable(2, 1);

% functions
F = x(1)^2 + x(2);
G = [2 - x(1); x(1) - 5; x(1) - 10];
H = x(2) - 5;

% solve
G_con = G <= 0;
H_con = H == 0;
opti.subject_to(G_con);
opti.subject_to(H_con);
opti.minimize(F);
opti.solver('ipopt', struct, struct('tol', 1e-1));
sol = opti.solve();

x_opt = sol.value(x); 
lam_opt = sol.value(opti.lam_g);


%% KKT conditions
kkts = {};

% stationarity
% jacobian(G, x) is (ng x nx)
dL = jacobian(F, x)' + jacobian(G, x)' * lam_opt(1:end-1) +  jacobian(H, x)' * lam_opt(end);
dL = MX2double(casadi.substitute(dL, x, sol.value(x)));
dL2 = jacobian(F + lam_opt(1:end-1)' * G + lam_opt(end)' * H, x);
dL2 = MX2double(casadi.substitute(dL2, x, sol.value(x)));
kkts{1} = dL; % should be zero

% primal feasibility
kkts{2} = sol.value(G); % should be non-positive
kkts{3} = sol.value(H); % should be zero

% dual feasibility 
kkts{4} = lam_opt(1:end-1); % should be non-negative

% complementary slackness
kkts{5} = lam_opt(1:end-1) .* sol.value(G); % should be zero


%% local
function v = MX2double(MX)
    v = full(evalf(MX));
end
