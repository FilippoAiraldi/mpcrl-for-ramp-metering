clc



%% plotting variables
% if no variables, load from file
if isempty(who())
    warning('off');
    load data\20220321_172841_data.mat
    warning('on');
end

% plotting step (to reduce number of datapoints to be plot)
step = 1;



%% Details
delimiter = ''; %'------------------------';
Table = {... % all entries must be strings

    % run details
    'RUN', delimiter; ...
    'name', runname;...
    'episodes', episodes; ...
    'tot exec time', duration(0, 0, exec_time_tot, 'Format', 'hh:mm:ss');...
    'mean ep exec time', ...
        duration(0, 0, mean(exec_times), 'Format', 'hh:mm:ss');

    % MPC details
    'MPC', delimiter; ...
    'Np/Nc/M', sprintf('%i/%i/%i', Np, Nc, M); ...
    'type cost initial', Vcost.name_out(); ...
    'type cost stage', Lcost.name_out(); ...
    'type cost terminal', Tcost.name_out(); ...
    'max iter', solver_opts.max_iter; ... 
    'rate var penalty weight', rate_var_penalty; ...
    'explor. perturbation mag.', perturb_mag; ...
    'max queue', max_queue; ...

    % RL details
    'RL', delimiter; ...
    'discount', discount; ...
    'learning rate', lr; ...
    'constraint violation penalty', con_violation_penalty; ...

    % learning outcomes
    'LEARNING', delimiter; ...
    'update frequency (iter)', rl_update_freq; ...
    'a (true)', sprintf('%7.3f', true_pars.a);...
    'v_free (true/initial/final)', sprintf('%7.3f / %7.3f / %7.3f', ...
        true_pars.v_free, rl_pars.v_free(1), rl_pars.v_free(end));...
    'rho_crit (true/initial/final)', sprintf('%7.3f / %7.3f / %7.3f', ...
        true_pars.rho_crit, rl_pars.rho_crit(1), rl_pars.rho_crit(end));...
    };
warning('still need to add for each learnable param the initial and final value')
    
Table = string(Table);
width = max(arrayfun(@(x) strlength(x), Table(:, 1))) + 4;
s = '';
for i = 1:size(Table, 1)
    if all(isstrprop(Table(i, 1),'upper'))
        if i > 1
            s = append(s, '\n');
        end
        delimiter = ' ';
    else
        delimiter = '-';
    end
    spaces = repmat(delimiter, 1, width - 2 - strlength(Table(i, 1)));
    s = append(s, Table(i, 1), ' ', spaces, ' ', Table(i, 2), '\n');
end
fprintf(s)



%% plotting
step = min(step, M);

% create figure
figure;
tiledlayout(4, 4, 'Padding', 'none', 'TileSpacing', 'compact')
sgtitle(runname, 'Interpreter', 'none')


% plot time-based quantities
t_tot = (0:(episodes * K - 1)) * T;
origins_tot = structfun(@(x) cell2mat(x), origins, 'UniformOutput', false);
links_tot = structfun(@(x) cell2mat(x), links, 'UniformOutput', false);
ax = matlab.graphics.axis.Axes.empty;

ax(1) = nexttile(1);
plot(t_tot(1:step:end), links_tot.speed(:, 1:step:end)');
hlegend(1) = legend('v_{L1}', 'v_{L2}', 'v_{L3}');
ylabel('speed (km/h)')

ax(2) = nexttile(2);
plot(t_tot(1:step:end), links_tot.flow(:, 1:step:end)')
hlegend(2) = legend('q_{L1}', 'q_{L2}', 'q_{L3}');
ylabel('flow (veh/h)')

ax(3) = nexttile(5);
plot(t_tot(1:step:end), links_tot.density(:, 1:step:end)', '-')
hlegend(3) = legend('\rho_{L1}', '\rho_{L2}', '\rho_{L3}');
ylabel('density (veh/km)')

ax(4) = nexttile(6);
slack_tot = mean(cell2mat(slack), 1);
plot(linspace(t_tot(1), t_tot(end), length(slack_tot)), slack_tot);
ylabel('slack \sigma')

ax(5) = nexttile(9); 
plot(t_tot(1:step:end), origins_tot.demand(:, 1:step:end)')
hlegend(5) = legend('d_{O1}', 'd_{O2}');
ylabel('origin demand (veh/h)')
% ax(5).YLim(2) = 4000;

ax(6) = nexttile(10); hold on
plot(t_tot(1:step:end), origins_tot.queue(:, 1:step:end)')
plot([t_tot(1), t_tot(end)], [max_queue, max_queue], '-.k')
hold off
hlegend(6) = legend('\omega_{O1}', '\omega_{O2}', 'max \omega_{O2}');
ylabel('queue length (veh)')

ax(7) = nexttile(13); 
plot(t_tot(1:step:end), origins_tot.flow(:, 1:step:end)')
hlegend(7) = legend('q_{O1}', 'q_{O2}');
ylabel('origin flow (veh/h)')

ax(8) = nexttile(14); hold on,
ax(8).ColorOrderIndex = 2;
stairs(t_tot(1:step:end), origins_tot.rate(:, 1:step:end)')
hlegend(8) = legend('r_{O2}');
ylabel('metering rate')

ax(9) = nexttile(11);
L_tot = full(Lrl(origins_tot.queue, links_tot.density));
plot(t_tot(1:step:end), L_tot(:, 1:step:end))
ylabel('L')

linkaxes(ax, 'x')
for i = 1:length(ax)
    xlabel(ax(i), 'time (h)')
    line(ax(i), repmat((1:episodes - 1) * Tfin, 2, 1), [0, ax(i).YLim(2)], ...
        'Color', '#686a70', 'LineStyle', ':', 'LineWidth', 0.75)
%     hold(ax(i), 'on')
%     plot(ax(i), (1:episodes) * Tfin, [0, ax(i).YLim(2)], ':k', 'LineWidth', 0.25)
%     hold(ax(i), 'off')
    if i <= length(hlegend) && isa(hlegend(i), 'matlab.graphics.illustration.Legend')
        hlegend(i).String = hlegend(i).String(1:end-episodes+1);
    end
    ax(i).YLim(1) = 0;
end


% plot episode-based quantities
ax = matlab.graphics.axis.Axes.empty;

ax(1) = nexttile(3, [1, 2]);
performance = arrayfun(@(ep) full(sum(Lrl(origins.queue{ep}, links.density{ep}))), 1:episodes);
performance_only_tts = arrayfun(@(ep) full(sum(TTS(origins.queue{ep}, links.density{ep}))), 1:episodes);
yyaxis left
plot(linspace(0, episodes, length(performance)), performance)
% stairs(linspace(0, episodes, length(performance) + 1), [performance, performance(end)])
% ax(1).YLim(1) = 0;
ylabel('J(\pi)')
yyaxis right
plot(linspace(0, episodes, length(performance_only_tts)), performance_only_tts)
% stairs(linspace(0, episodes, length(performance_only_tts) + 1), [performance_only_tts, performance_only_tts(end)])
ylabel('TTS(\pi)')
% bar(0.5:1:(episodes-0.5), [performance_only_tts', (performance - performance_only_tts)'], 'stacked')
% ylabel('J(\pi)')


ax(2) = nexttile(7, [1, 2]);
td_error_tot = cell2mat(td_error);
plot(linspace(0, episodes, length(td_error_tot)), td_error_tot, 'o', 'MarkerSize', 2)
ylabel('TD error \tau')

ax(3) = nexttile(15); hold on
stairs(linspace(0, episodes, length(rl_pars.v_free)), rl_pars.v_free)
stairs(linspace(0, episodes, length(rl_pars.rho_crit)), rl_pars.rho_crit)
ax(3).ColorOrderIndex = 1;
plot([0, episodes], [true_pars.v_free, true_pars.v_free], '--')
plot([0, episodes], [true_pars.rho_crit, true_pars.rho_crit], '--')
legend('v_{free}', '\rho_{crit}')
hold off
ylabel('v_{free}, \rho_{crit}')

ax(4) = nexttile(12, [2, 1]); hold on
weights = fieldnames(rl_pars);
legendStrings = {};
for name = weights(3:end)' % first 2 are v_free and rho_crit
    weight = rl_pars.(name{1});
    for i = 1:size(weight, 1)
        w = weight(i, :);
        stairs(linspace(0, episodes, length(w)), w)
        legendStrings{end + 1} = append(name, '_', string(i));
    end
end
hold off
legend(legendStrings{:}, 'interpreter', 'none')
ylabel('weights')

linkaxes(ax, 'x')
for i = 1:length(ax)
    xlabel(ax(i), 'episode')
end
