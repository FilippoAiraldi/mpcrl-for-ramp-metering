clearvars, clc, close all

% data
eps = 1e-4;
rho_max = 180;
true_pars = struct('a', 1.867, 'v_free', 102, 'rho_crit', 33.5);
a = true_pars.a * 1.3;
v_free = true_pars.v_free * 1.3;
rho_crit = true_pars.rho_crit * 0.7;

% real curve
rho = linspace(0, rho_max, 1e3);
v_real = Veq(rho, v_free, a, rho_crit, eps);

% approximation via Matlab
Veq_approx = get_Veq_approx();
p = lsqcurvefit(...
    @(p, rho_i) full(Veq_approx(rho_i, v_free, p(1), p(2))), ...
    [-1, 1], ...    % inital guess
    rho, ...
    v_real, ...
    [-inf, 0], ...  % lb
    [0, v_free]);   % ub
v_approx_matlab = Veq_approx(rho, v_free, p(1), p(2));

% plot comparison
plot(rho, v_real, rho, full(v_approx_matlab))

% NOTES: when learning, height2 should never be exactly zero, there should
% be a minimum beneath which it cannot go


%% local functions
function V = Veq(rho, v_free, a, rho_crit, eps)
    V = v_free * exp((-1 / a) * ((rho / rho_crit) + eps).^a);
end

function f = get_Veq_approx()
    rho = casadi.SX.sym('rho', 1, 1);
    v_free = casadi.SX.sym('v_free', 1, 1);
    slope1 = casadi.SX.sym('slope1', 1, 1);
    height2 = casadi.SX.sym('height2', 1, 1);
    f = casadi.Function('Veq_approx', ...
        {rho, v_free, slope1, height2}, ...
        {max(slope1 * rho + v_free, height2)}, ...
        {'rho', 'v_free', 'slope1', 'height2'}, {'v'});
end
