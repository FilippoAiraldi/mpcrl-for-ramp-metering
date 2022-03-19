clc



%% plotting variables
% if no variables, load from file
if isempty(who())
    warning('off');
    load 20220319_124248_data.mat
    warning('on');
end

% plotting step (to reduce number of datapoints to be plot)
step = 3;



%% Details
delimiter = ''; %'------------------------';
Table = {... % all entries must be strings

    % run details
    'RUN', delimiter; ...
    'name', runname;...
    'episodes', episodes; ...
    'tot exec time', duration(0, 0, exec_time_tot, 'Format', 'hh:mm:ss');...
    'mean ep exec time', ...
        duration(0, 0, mean(cell2mat(exec_times)), 'Format', 'hh:mm:ss');

    % types of costs
    'RLMPC', delimiter; ...
    'Np/Nc/M', sprintf('%i/%i/%i', Np, Nc, M); ...
    'type cost initial', Vcost.name_out(); ...
    'type cost stage', Lcost.name_out(); ...
    'type cost terminal', Tcost.name_out(); ...
    'max iter', solver_opts.max_iter; ... 
    'rate var penalty', rate_var_penalty; ...
    'max queue', max_queue; ...
    'discount', discount; ...
    'learning rate', lr; ...

    % learning
    'LEARNING', delimiter; ...
    'updates at (iter/time)', sprintf('%s / %s', ...
        mat2str(rl_update_at), mat2str(rl_update_at / K * Tfin)); ...
    'a (true)', sprintf('%7.3f', true_pars.a);...
    'v_free (true/initial/final)', sprintf('%7.3f / %7.3f / %7.3f', ...
        true_pars.v_free, rl_pars.v_free(1), rl_pars.v_free(end));...
    'rho_crit (true/initial/final)', sprintf('%7.3f / %7.3f / %7.3f', ...
        true_pars.rho_crit, rl_pars.rho_crit(1), rl_pars.rho_crit(end));...
    };
    
Table = string(Table);
width = max(arrayfun(@(x) strlength(x), Table(:, 1))) + 6;
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
% create figure
figure;
tiledlayout(4, 4, 'Padding', 'none', 'TileSpacing', 'compact')
sgtitle(runname, 'Interpreter', 'none')


% plot traffic quantities
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

ax(4) = nexttile(9); 
plot(t_tot(1:step:end), repmat(D(:, 1:step:end), 1, episodes)')
hlegend(4) = legend('d_{O1}', 'd_{O2}');
ylabel('origin demand (veh/h)')
% ax(5).YLim(2) = 4000;

ax(5) = nexttile(10); hold on
plot(t_tot(1:step:end), origins_tot.queue(:, 1:step:end)')
plot([t_tot(1), t_tot(end)], [max_queue, max_queue], '-.k')
hold off
hlegend(5) = legend('\omega_{O1}', '\omega_{O2}', 'max \omega_{O2}');
ylabel('queue length (veh)')

ax(6) = nexttile(13); 
plot(t_tot(1:step:end), origins_tot.flow(:, 1:step:end)')
hlegend(6) = legend('q_{O1}', 'q_{O2}');
ylabel('origin flow (veh/h)')

ax(7) = nexttile(14); hold on,
ax(7).ColorOrderIndex = 2;
stairs(t_tot(1:step:end), origins_tot.rate(:, 1:step:end)')
hlegend(7) = legend('r_{O2}');
ylabel('metering rate')

linkaxes(ax, 'x')
for i = 1:length(ax)
    xlabel(ax(i), 'time (h)')
    line(ax(i), repmat((1:episodes - 1) * Tfin, 2, 1), [0, ax(i).YLim(2)], ...
        'Color', '#686a70', 'LineStyle', ':', 'LineWidth', 0.75)
%     hold(ax(i), 'on')
%     plot(ax(i), (1:episodes) * Tfin, [0, ax(i).YLim(2)], ':k', 'LineWidth', 0.25)
%     hold(ax(i), 'off')
    hlegend(i).String = hlegend(i).String(1:end-episodes+1);
    ax(i).YLim(1) = 0;
end


% plot learning quantities
ax = matlab.graphics.axis.Axes.empty;

ax(1) = nexttile(3, [1, 2]);
plot(1:episodes, cellfun(@(x) sum(x), objective))
ylabel('objective')

ax(2) = nexttile(7);
slack_tot = mean(cell2mat(slack), 1);
plot(linspace(1, episodes, length(slack_tot)), slack_tot)
ylabel('slack')

ax(3) = nexttile(8);
td_error_tot = cell2mat(objective);
plot(linspace(1, episodes, length(td_error_tot)), td_error_tot)
ylabel('TD error')

% tiles 11, 12, 15, 16 for showing convergences of learned parameters

% ax(3) = nexttile(11);
% plot()
% plot v_free and rho_crit parameters as they are updated. Plot horizontal
% line with the true parameters

linkaxes(ax, 'x')
for i = 1:length(ax)
    xlabel(ax(i), 'episode')
end