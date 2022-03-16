clc, clearvars, close all


%% logging and saving
runname = datestr(datetime,'yyyymmdd_HHMMSS');
log_filename = strcat(runname, '_log.txt');
result_filename = strcat(runname, '_data.mat');
diary(log_filename)   


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
v_free = [102, 130];            % free flow speed (km/h)
rho_crit = [33.5, 27];          % critical capacity (veh/km/lane)


%% Disturbances
% original disturbance
% d1 = util.create_profile(t, [0, .25, 1, 1.25], [1000, 3500, 3500, 1000]);
% d2 = util.create_profile(t, [.25, .37, .62, .75], [500, 1750, 1750, 500]);

% result 1 disturbance - small difference in cost, no slack
% d1 = util.create_profile(t, [0, .3, .95, 1.25], [1000, 3100, 3100, 1000]);
% d2 = util.create_profile(t, [.15, .32, .57, .75], [500, 1800, 1800, 500]);

% result 2 disturbance - more difference in cost, some slack
d1 = util.create_profile(t, [0, .3, .95, 1.25], [1000, 3150, 3150, 1000]);
d2 = util.create_profile(t, [.15, .32, .57, .75], [500, 1800, 1800, 500]);

% non-piecewise disturbances
% d1 = 2500 * (math.sigmoid(20 * (t - 0.15)) - math.sigmoid(20 * (t - 1.1))) + 1000;
% d2 = 1250 * (math.sigmoid(40 * (t - 0.225)) - math.sigmoid(40 * (t - 0.66))) + 500;

% assemble and plot disturbances
D = [d1; d2];
% plot(t, d1, t, d2),
% legend('O1', 'O2'), xlabel('time (h)'), ylabel('demand (h)')
% ylim([0, 4000])


%% MPC
% create casadi function to run dynamics with any parameter
F = util.f2casadiF(T, L, lanes, C2, rho_max, tau, delta, eta, kappa);

% build the two mpcs
Np = 7;
Nc = 3;
M = 6;
MPCs(1) = metanet.MPC(Np, Nc, M, F, a(1), v_free(1), rho_crit(1), T, L, lanes);
MPCs(2) = metanet.MPC(Np, Nc, M, F, a(2), v_free(2), rho_crit(2), T, L, lanes);


%% Simulation
% create containers 
w = {nan(size(MPCs(1).vars.w, 1), K + 1), nan(size(MPCs(2).vars.w, 1), K + 1)};
q_o = {nan(size(MPCs(1).vars.w, 1), K), nan(size(MPCs(2).vars.w, 1), K)};
r = {nan(size(MPCs(1).vars.r, 1), K), nan(size(MPCs(2).vars.r, 1), K)};
q = {nan(size(MPCs(1).vars.v, 1), K), nan(size(MPCs(2).vars.v, 1), K)};
rho = {nan(size(MPCs(1).vars.rho, 1), K + 1), nan(size(MPCs(2).vars.rho, 1), K + 1)};
v = {nan(size(MPCs(1).vars.v, 1), K + 1), nan(size(MPCs(2).vars.v, 1), K + 1)};
slack = {nan(size(MPCs(1).vars.slack, 2), K), nan(size(MPCs(1).vars.slack, 2), K)};
% slack = {nan(1, K), nan(1, K)};
objectives = {nan(1, K), nan(1, K)}; 

% initial conditions
w{1}(:, 1) = zeros(2, 1);
rho{1}(:, 1) = [4.98586, 5.10082, 7.63387]';
v{1}(:, 1) = [100.297, 98.0923, 98.4106]';
r{1}(:, 1) = 1;
w{2}(:, 1) = w{1}(:, 1);
rho{2}(:, 1) = rho{1}(:, 1);
v{2}(:, 1) = v{1}(:, 1);
r{2}(:, 1) = r{1}(:, 1);
w_last = {repmat(w{1}(:, 1), 1, M * Np + 1),     repmat(w{2}(:, 1), 1, M * Np + 1)};
rho_last = {repmat(rho{1}(:, 1), 1, M * Np + 1), repmat(rho{2}(:, 1), 1, M * Np + 1)};
v_last = {repmat(v{1}(:, 1), 1, M * Np + 1),     repmat(v{2}(:, 1), 1, M * Np + 1)};
r_last = {repmat(r{1}(:, 1), 1, Nc),             repmat(r{2}(:, 1), 1, Nc)};

%  loop
start_time = tic;
for k = 1:K
    any_error = false;
    if mod(k, M) == 1
        % predict disturbances
        if k <= K - M * Np + 1
            d = D(:, k:k + M * Np - 1);
        else
            d = [D(:, k:end), repmat(D(:, end), 1, M * Np - 1 - K + k)];
        end

        % run MPCs
        for i = 1:2
            [w_last{i}, rho_last{i}, v_last{i}, r_last{i}, info] = MPCs(i).solve(...
                d, w{i}(:, k), rho{i}(:, k), v{i}(:, k), r_last{i}(1), ...
                util.build_input(w{i}(:, k), w_last{i}, M), ...
                util.build_input(rho{i}(:, k), rho_last{i}, M), ...
                util.build_input(v{i}(:, k), v_last{i}, M), ...
                [r_last{i}(2:end), r_last{i}(end)]);
%             [w_last{i}, rho_last{i}, v_last{i}, r_last{i}, info] = MPCs(i).solve(...
%                 d, w{i}(:, k), rho{i}(:, k), v{i}(:, k), r_last{i}(1), ...
%                 w_last{i}, rho_last{i}, v_last{i}, r_last{i});
            if isfield(info, 'error')
                any_error = true; 
                util.logging(k, K, t(k), toc(start_time), sprintf('(%i) %s', i, info.error));
            end

            % save infos
            slack{i}(:, k:k + M - 1) = repmat(info.slack', 1, M);
            objectives{i}(:, k:k + M - 1) = info.f;
        end
    end

    % report progress
    if ~any_error && mod(k, 4 * M) == 1
        util.logging(k, K, t(k), toc(start_time));
    end

    for i = 1:2
        % assign metering rate
        r{i}(:, k) = r_last{i}(:, 1);
    
        % step simulation - always with true parameters
        [q_o_, w_, q_, rho_, v_] = F(w{i}(:, k), rho{i}(:, k), v{i}(:, k), r{i}(:, k), ...
            D(:, k), a(1), v_free(1), rho_crit(1));
        q_o{i}(:, k) = full(q_o_);
        w{i}(:, k + 1) = full(w_);
        q{i}(:, k) = full(q_);
        rho{i}(:, k + 1) = full(rho_);
        v{i}(:, k + 1) = full(v_);
    end
end
exec_time = toc(start_time);
fprintf('Execution time is %f\n', exec_time)


%% Saving

% compute cost
TTS = zeros(size(MPCs));
for i = 1:2
    TTS(i) = metanet.TTS(w{i}, rho{i}, T, L, lanes);
    J = sum(objectives{i}, 'all');
    fprintf('(%i) TTS = %f, J = %f\n', i, TTS(i), J)
end
diary off

% perform save
time = t;
origin_flow = q_o;
origin_queue = w;
origin_demand = D;
origin_rate = r;
link_flow = q;
link_density = rho;
link_speed = v;
warning('off', 'all');
save(result_filename, 'exec_time', 'MPCs', 'TTS', 'objectives', 'time', ...
    'origin_flow', 'origin_queue', 'origin_demand', 'origin_rate', ...
    'link_flow', 'link_density', 'link_speed', 'slack');
warning('on', 'all');



%% Plotting
tiledlayout(5, 2, 'Padding', 'none', 'TileSpacing', 'compact')
% sgtitle(filename)

ax(1) = nexttile; hold on,
plot(t, v{1}(:, 1:end-1), '-')
ax(1).ColorOrderIndex = 1;
plot(t, v{2}(:, 1:end-1), '--')
legend('v_{L1}', 'v_{L2}', 'v_{L3}')
ylabel('speed (km/h)')

ax(2) = nexttile; hold on,
plot(t, q{1}, '-')
ax(2).ColorOrderIndex = 1;
plot(t, q{2}, '--')
legend('q_{L1}', 'q_{L2}', 'q_{L3}')
ylabel('flow (veh/h)')

ax(3) = nexttile; hold on,
plot(t, rho{1}(:, 1:end-1), '-')
ax(3).ColorOrderIndex = 1;
plot(t, rho{2}(:, 1:end-1), '--')
legend('\rho_{L1}', '\rho_{L2}', '\rho_{L3}')
ylabel('density (veh/km)')

ax(4) = nexttile;
ax(4).Visible = 'off';

ax(5) = nexttile; 
plot(t, D)
legend('d_{O1}', 'd_{O2}')
ylabel('origin demand (veh/h)')
ax(5).YLim(2) = 3500;

ax(6) = nexttile; hold on,
plot(t, w{1}(:, 1:end-1), '-')
ax(6).ColorOrderIndex = 1;
plot(t, w{2}(:, 1:end-1), '--')
plot([t(1), t(end)], [100, 100], '-.k')
legend('\omega_{O1}', '\omega_{O2}', '', '', '\omega_{O2} constr.')
ylabel('queue length (veh)')

ax(7) = nexttile; hold on,
plot(t, q_o{1}, '-')
ax(7).ColorOrderIndex = 1;
plot(t, q_o{2}, '--')
legend('q_{O1}', 'q_{O2}')
ylabel('origin flow (veh/h)')

ax(8) = nexttile; hold on,
ax(8).ColorOrderIndex = 2;
stairs(t, r{1}, '-')
ax(8).ColorOrderIndex = 2;
stairs(t, r{2}, '--')
legend('r_{O2}')
ylabel('metering rate')

ax(9) =  nexttile; hold on,
stairs(t, objectives{1}, '-')
ax(9).ColorOrderIndex = 1;
stairs(t, objectives{2}, '--')
ylabel('J_{MPC} (veh \cdot h)')

ax(10) =  nexttile; hold on,
plot(t, slack{1}, '-')
ax(10).ColorOrderIndex = 1;
plot(t, slack{2}, '--')
ylabel({'slack variable', '(\omega_{O2} constraint)'})

linkaxes(ax, 'x')
for i = 1:length(ax)
    xlabel(ax(i), 'time (h)')
    ax(i).YLim(1) = 0;
end
