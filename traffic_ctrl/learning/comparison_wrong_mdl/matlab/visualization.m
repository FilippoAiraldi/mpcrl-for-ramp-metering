% clc, clearvars, close all %#ok<*SAGROW> 


% filename = 'data\base.mat';
% filename = 'data\slack_1d_weight_10.mat';
filename = 'result_good2.mat';
step = 2;


%% load
warning('off', 'all');
load(filename)
warning('on', 'all');


%% create table
for i = 1:2
    a(i) = MPCs(i).a;
    v_free(i) = MPCs(i).v_free; 
    rho_crit(i) = MPCs(i).rho_crit;
    exec_time_tot(i) = duration(0, 0, exec_time, 'Format', 'hh:mm:ss');
end
T = table(a', v_free', rho_crit', TTS', [sum(objectives{1}); sum(objectives{2})], exec_time_tot', ...
    'VariableNames', {'a','v_free', 'rho_crit', 'TTS', 'J', 'exec_time (total)'});
disp(T)


%% plot
figure;
tiledlayout(5, 2, 'Padding', 'none', 'TileSpacing', 'compact')
% sgtitle(filename)

ax(1) = nexttile; hold on,
plot(time(1:step:end), link_speed{1}(:, 1:step:end-1)', '-')
ax(1).ColorOrderIndex = 1;
plot(time(1:step:end), link_speed{2}(:, 1:step:end-1)', '--')
legend('v_{L1}', 'v_{L2}', 'v_{L3}')
ylabel('speed (km/h)')

ax(2) = nexttile; hold on,
plot(time(1:step:end), link_flow{1}(:, 1:step:end)', '-')
ax(2).ColorOrderIndex = 1;
plot(time(1:step:end), link_flow{2}(:, 1:step:end)', '--')
legend('q_{L1}', 'q_{L2}', 'q_{L3}')
ylabel('flow (veh/h)')

ax(3) = nexttile; hold on,
plot(time(1:step:end), link_density{1}(:, 1:step:end-1)', '-')
ax(3).ColorOrderIndex = 1;
plot(time(1:step:end), link_density{2}(:, 1:step:end-1)', '--')
legend('\rho_{L1}', '\rho_{L2}', '\rho_{L3}')
ylabel('density (veh/km)')

ax(4) = nexttile;
ax(4).Visible = 'off';

ax(5) = nexttile; 
plot(time(1:step:end), origin_demand(:, 1:step:end)')
legend('d_{O1}', 'd_{O2}')
ylabel('origin demand (veh/h)')
ax(5).YLim(2) = 3500;

ax(6) = nexttile; hold on,
plot(time(1:step:end), origin_queue{1}(:, 1:step:end-1)', '-')
ax(6).ColorOrderIndex = 1;
plot(time(1:step:end), origin_queue{2}(:, 1:step:end-1)', '--')
plot([time(1), time(end)], [100, 100], '-.k')
legend('\omega_{O1}', '\omega_{O2}', '', '', '\omega_{O2} constr.')
ylabel('queue length (veh)')

ax(7) = nexttile; hold on,
plot(time(1:step:end), origin_flow{1}(:, 1:step:end)', '-')
ax(7).ColorOrderIndex = 1;
plot(time(1:step:end), origin_flow{2}(:, 1:step:end)', '--')
legend('q_{O1}', 'q_{O2}')
ylabel('origin flow (veh/h)')

ax(8) = nexttile; hold on,
ax(8).ColorOrderIndex = 2;
stairs(time(1:step:end), origin_rate{1}(:, 1:step:end)', '-')
ax(8).ColorOrderIndex = 2;
stairs(time(1:step:end), origin_rate{2}(:, 1:step:end)', '--')
legend('r_{O2}')
ylabel('metering rate')

ax(9) =  nexttile; hold on,
stairs(time(1:step:end), objectives{1}(1:step:end), '-')
ax(9).ColorOrderIndex = 1;
stairs(time(1:step:end), objectives{2}(1:step:end), '--')
ylabel('J_{MPC} (veh \cdot h)')

ax(10) =  nexttile; hold on,
stairs(time(1:step:end), slack{1}(:, 1:step:end)', '-')
ax(10).ColorOrderIndex = 1;
stairs(time(1:step:end), slack{2}(:, 1:step:end)', '--')
ylabel({'slack variable', '(\omega_{Ow} constraint)'})

linkaxes(ax, 'x')
for i = 1:length(ax)
    xlabel(ax(i), 'time (h)')
    ax(i).YLim(1) = 0;
end
