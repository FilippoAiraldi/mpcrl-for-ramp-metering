clc, clearvars, close all


%% Model
% simulation
Tstage = 1.5;                   % simulation time per stage (h)
stages = 1;                     % number of repetitions basically
Tfin = Tstage * stages;         % final simulation time (h)
T = 10 / 3600;                  % simulation step size (h)
t = 0:T:Tfin - T;               % time vector (h)
K = numel(t);                   % simulation steps
Kstage = K / stages;            % simulation steps per stage

% segments
L = 1;                          % length of links (km)
lanes = 2;                      % lanes per link (adim)

% on-ramp O2
C2 = 2000;                      % on-ramp capacity (veh/h/lane)
max_queue = 100;

% model parameters
tau = 18 / 3600;                % model parameter (s)
kappa = 40;                     % model parameter (veh/km/lane)
eta = 60;                       % model parameter (km^2/lane)
rho_max = 180;                  % maximum capacity (veh/km/lane)
delta = 0.0122;                 % merging phenomenum parameter

% true and wrong model parameters
a = [1.867, 2.111];             % model parameter (adim)
v_free = [102, 120];            % free flow speed (km/h)
rho_crit = [33.5, 27];          % critical capacity (veh/km/lane)

% save everything in a struct
config.L = L;
config.lanes = lanes;
config.v_free = v_free(1);
config.rho_crit = rho_crit(1);
config.rho_max = rho_max;
config.a = a(1);
config.tau = tau;
config.eta = eta;
config.kappa = kappa;
config.delta = delta;
config.C2 = C2;
config.T = T;


%% Disturbances
d1 = metanet.create_profile(t, [0, .25, 1, 1.25], [1000, 3500, 3500, 1000]);
d2 = metanet.create_profile(t, [.25, .37, .62, .75], [500, 1750, 1750, 500]);
D = [d1; d2];

% plot(t, d1, t, d2),
% legend('O1', 'O2'), xlabel('time (h)'), ylabel('demand (h)')
% ylim([0, 4000])


%% MPC
% mpc parameters
Np = 7;
Nc = 3;
M = 6;
opti = casadi.Opti();

% create variables
opti_var.w = opti.variable(2, M * Np + 1);      % queues
opti_var.rho = opti.variable(3, M * Np + 1);    % densities
opti_var.v = opti.variable(3, M * Np + 1);      % speeds
opti_var.r = opti.variable(1, Nc);              % ramp metering
opti_par.d = opti.parameter(2, M * Np);         % demands
opti_par.w0 = opti.parameter(2, 1);
opti_par.rho0 = opti.parameter(3, 1);
opti_par.v0 = opti.parameter(3, 1);
opti_par.r_last = opti.parameter(1, 1);
% w_opti = casadi.SX.sym('w', 2, M * Np + 1);
% rho_opti = casadi.SX.sym('rho', 3, M * Np + 1);
% v_opti = casadi.SX.sym('v', 3, M * Np + 1);
% r_opti = casadi.SX.sym('r', 1, Nc);  
% d_opti = casadi.SX.sym('d', 2, M * Np);
% w0_opti = casadi.SX.sym('w0', 2, 1);
% rho0_opti = casadi.SX.sym('rho0', 3, 1);
% v0_opti = casadi.SX.sym('v0', 3, 1);
% r_last_opti = casadi.SX.sym('r_last', 1, 1);

% create casadi function
F = metanet.f2casadiF(config);

% cost to minimize
cost = metanet.TTS(opti_var.w, opti_var.rho, T, L, lanes) + ...
        0.4 * metanet.input_variability_penalty(opti_par.r_last, opti_var.r);
opti.minimize(cost)

% constraints on domains
opti.subject_to(0.2 <= opti_var.r <= 1) %#ok<CHAIN> 
opti.subject_to(opti_var.w(:) >= 0)
opti.subject_to(opti_var.rho(:) >= 0)
opti.subject_to(opti_var.v(:) >= 0)

% constraints on initial conditions
opti.subject_to(opti_var.w(:, 1) == opti_par.w0)
opti.subject_to(opti_var.v(:, 1) == opti_par.v0)
opti.subject_to(opti_var.rho(:, 1) == opti_par.rho0)

% constraints on state evolution
r_exp = [repelem(opti_var.r, 6), repelem(opti_var.r(end), M * (Np - Nc))'];
for k = 1:M * Np
    [~, w_next, ~, rho_next, v_next] = F(opti_var.w(:, k), ...
        opti_var.rho(:, k), opti_var.v(:, k), r_exp(:, k), opti_par.d(:, k));
    opti.subject_to(opti_var.w(:, k + 1) == w_next)
    opti.subject_to(opti_var.rho(:, k + 1) == rho_next)
    opti.subject_to(opti_var.v(:, k + 1) == v_next)
end

% set solver for opti
plugin_opts = struct('expand', false, 'print_time', false);
solver_opts = struct('print_level', 0, ...
    'max_iter', 6e3, 'max_cpu_time', 1e1);
opti.solver('ipopt', plugin_opts, solver_opts)


%% Simulation
% create containers
w = nan(size(opti_var.w, 1), K + 1);
q_o = nan(size(opti_var.w, 1), K);
r = nan(size(opti_var.r, 1), K);
q = nan(size(opti_var.rho, 1), K);
rho = nan(size(opti_var.rho, 1), K + 1);
v = nan(size(opti_var.v, 1), K + 1);

% initial conditions
w(:, 1) = zeros(2, 1);
rho(:, 1) = [4.98586, 5.10082, 7.63387]';
v(:, 1) = [100.297, 98.0923, 98.4106]';
r(:, 1) = 0.5;
w_last = repmat(w(:, 1), 1, M * Np + 1);
rho_last = repmat(rho(:, 1), 1, M * Np + 1);
v_last = repmat(v(:, 1), 1, M * Np + 1);
r_last = repmat(r(:, 1), 1, Nc);


% override rate from a python file
use_python_rate = false;
if use_python_rate
    r_python = load('rate_python.mat').r;
end


%  loop
for k = 1:K
    if mod(k, M) == 1 && ~use_python_rate
        % predict disturbances
        if k <= K - M * Np + 1
            d = D(:, k:k + M * Np - 1);
        else
            d = [D(:, k:end), repmat(D(:, end), 1, M * Np - 1 - K + k)];
        end
        assert(isequal(size(d), [2, 42]))

        % run MPC
        [~, w_last, rho_last, v_last, r_last, info] = MCP(...
            opti, opti_var, opti_par, ...
            d, w(:, k), rho(:, k), v(:, k), r_last(1), ...
            metanet.build_input(w(:, k), w_last, M), ...
            metanet.build_input(rho(:, k), rho_last, M), ...
            metanet.build_input(v(:, k), v_last, M), ...
            [r_last(2:end), r_last(end)]);
        if isfield(info, 'error')
            fprintf('error - %i/%2.2f%%: %s\n', k, k / K * 100, info.error)
        elseif mod(k, 4 * M) == 1
            fprintf('progress - %i/%2.2f%%\n', k, k / K * 100)
        end
    end

    if use_python_rate
        r(:, k) = r_python(:, k);
    else
        % assign metering rate
        r(:, k) = r_last(:, 1);
    end


    % step simulation
    [q_o_, w_, q_, rho_, v_] = F(w(:, k), rho(:, k), v(:, k), r(:, k), ...
        D(:, k));
    q_o(:, k) = full(q_o_);
    w(:, k + 1) = full(w_);
    q(:, k) = full(q_);
    rho(:, k + 1) = full(rho_);
    v(:, k + 1) = full(v_);
end


%% Plotting

save_rate = false;
if save_rate
    save rate_matlab.mat r
end

% compute cost
tts = metanet.TTS(w, rho, T, L, lanes);
fprintf('TTS = %f\n', tts)

% plot
tiledlayout(4, 2, 'Padding', 'none', 'TileSpacing', 'compact')

nexttile,
plot(t, v(:, 1:end-1))
legend('v_{L1}', 'v_{L2}', 'v_{L3}')
labels('time (h)', 'speed (km/h)')

nexttile,
plot(t, q)
legend('q_{L1}', 'q_{L2}', 'q_{L3}')
labels('time (h)', 'flow (veh/h)')

nexttile,
plot(t, rho(:, 1:end-1))
legend('\rho_{L1}', '\rho_{L2}', '\rho_{L3}')
labels('time (h)', 'density (veh/km)')

nexttile,
axis off

nexttile,
plot(t, D)
legend('d_{O1}', 'd_{O2}')
labels('time (h)', 'origin demand (veh/h)')

nexttile,
plot(t, w(:, 1:end-1))
legend('\omega_{O1}', '\omega_{O2}')
labels('time (h)', 'queue length (veh)')

nexttile,
plot(t, q_o)
legend('q_{O1}', 'q_{O2}')
labels('time (h)', 'origin flow (veh/h)')

nexttile,
plot(t, r)
legend('r_{O2}')
labels('time (h)', 'metering rate')



%% local functions
function [f, w_opt, rho_opt, v_opt, r_opt, info] = MCP(opti, opti_var, opti_par, d, w0, rho0, v0, r0, w_last, rho_last, v_last, r_last)
    % set parameters
    opti.set_value(opti_par.d, d);
    opti.set_value(opti_par.w0, w0);
    opti.set_value(opti_par.rho0, rho0);
    opti.set_value(opti_par.v0, v0);
    opti.set_value(opti_par.r_last, r0);

    % warm start
    opti.set_initial(opti_var.w, w_last);
    opti.set_initial(opti_var.rho, rho_last);
    opti.set_initial(opti_var.v, v_last);
    opti.set_initial(opti_var.r, r_last);

    % run solver
    try
        sol = opti.solve();
        info = struct();
        get_value = @(o) sol.value(o);
    catch ME1
        try
            stats = opti.debug.stats();
            info = struct('error', stats.return_status);
            get_value = @(o) opti.debug.value(o);
        catch ME2
            msg = ['error during handling of first exception.\nEx. 1:', ...
                    ME1,'\nEx. 2:', ME2];
            ME2 = addCause(ME2, MException('casadi', msg));
            rethrow(ME2)
        end   
    end 
    
    % get outputs
    f = get_value(opti.f);
    w_opt = get_value(opti_var.w);
    rho_opt = get_value(opti_var.rho);
    v_opt = get_value(opti_var.v);
    r_opt = get_value(opti_var.r);
end