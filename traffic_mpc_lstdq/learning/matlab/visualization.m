% clc
% clear all


%% plotting variables
% if no variables, load from file
if isempty(who())
    warning('off');
    load data\20220326_211358_data.mat
    warning('on');

    % if loading a checkpoint, fill missing variables
    if ~exist('exec_time_tot','var')
        exec_time_tot = nan;
        rl_pars = structfun(@(x) cell2mat(x), rl_pars, 'UniformOutput', false);
    end
end

% plotting step (to reduce number of datapoints to be plot)
step = 3;
plot_traffic = false;
plot_learning = true;
scaled_learned = true;



%% Details
delimiter = ''; %'------------------------';
Table = {... % all entries must be strings
    % run details
    'RUN', delimiter; ...
    'name', runname;...
    'episodes', sprintf('%i (executed %i)', episodes, ep); ...
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
    'max queues', sprintf('%i, %i', max_queue1, max_queue2); ...
    'epsilon', eps; ...

    % RL details
    'RL', delimiter; ...
    'discount', discount; ...
    'learning rate', lr; ...
    'constraint violation penalty', con_violation_penalty; ...

    % learning outcomes
    'LEARNING', delimiter; ...
    'update frequency (iter)', rl_update_freq; ...
    'a (true)', sprintf('%7.3f', true_pars.a);...
    'v_free (true/init/fin)', sprintf('%7.3f / %7.3f / %7.3f', ...
        true_pars.v_free, rl_pars.v_free(1), rl_pars.v_free(end));...
    'rho_crit (true/init/fin)', sprintf('%7.3f / %7.3f / %7.3f', ...
        true_pars.rho_crit, rl_pars.rho_crit(1), rl_pars.rho_crit(end));...
    };
weights = fieldnames(rl_pars);
for name = weights(3:end)' % first 2 are v_free and rho_crit
    weight = rl_pars.(name{1});
    for i = 1:size(weight, 1)
        w = weight(i, 1);
        Table = [Table; { sprintf('%s_%i (init/fin)', name{1}, i), ...
            sprintf('%7.3f / %7.3f', weight(i, 1), weight(i, end))}];
    end
end
 
Table = string(Table);
Table(ismissing(Table)) = 'NaN';
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
ep_tot = min(ep, episodes); % minimum of episode index and total episodes

% convert episode cells to one big array
t_tot = (0:(ep_tot * K - 1)) * T;
origins_tot = structfun(@(x) cell2mat(x), origins, 'UniformOutput', false);
links_tot = structfun(@(x) cell2mat(x), links, 'UniformOutput', false);

% traffic quantities figure
if plot_traffic
    figure;
    tiledlayout(4, 2, 'Padding', 'none', 'TileSpacing', 'compact')
    sgtitle(runname, 'Interpreter', 'none')
    ax = matlab.graphics.axis.Axes.empty;
    
    ax(1) = nexttile(1);
    plot(t_tot(1:step:end), links_tot.speed(:, 1:step:end)');
    hlegend(1) = legend('v_{L1}', 'v_{L2}', 'v_{L3}');
    ylabel('speed (km/h)')
    
    ax(2) = nexttile(2);
    plot(t_tot(1:step:end), links_tot.flow(:, 1:step:end)')
    hlegend(2) = legend('q_{L1}', 'q_{L2}', 'q_{L3}');
    ylabel('flow (veh/h)')
    
    ax(3) = nexttile(3);
    plot(t_tot(1:step:end), links_tot.density(:, 1:step:end)', '-')
    hlegend(3) = legend('\rho_{L1}', '\rho_{L2}', '\rho_{L3}');
    ylabel('density (veh/km)')
    
    ax(4) = nexttile(4);
    slack_tot = mean(cell2mat(slack), 1);
    plot(linspace(t_tot(1), t_tot(end), length(slack_tot)), slack_tot);
    ylabel('slack \sigma')
    
    ax(5) = nexttile(5); 
    if size(origins_tot.demand, 1) > 2
        plot(t_tot(1:step:end), (origins_tot.demand(:, 1:step:length(t_tot)) .* [1; 1; 50])')
        hlegend(5) = legend('d_{O1}', 'd_{O2}', 'd_{cong}\times50');
    else
        plot(t_tot(1:step:end), origins_tot.demand(:, 1:step:length(t_tot))')
        hlegend(5) = legend('d_{O1}', 'd_{O2}');
    end
    ylabel('origin demand (veh/h)')
    % ax(5).YLim(2) = 4000;
    
    ax(6) = nexttile(6); hold on
    plot(t_tot(1:step:end), origins_tot.queue(:, 1:step:end)')
    plot([t_tot(1), t_tot(end)], [max_queue1, max_queue1], '-.k')
    plot([t_tot(1), t_tot(end)], [max_queue2, max_queue2], '-.k')
    hold off
    hlegend(6) = legend('\omega_{O1}', '\omega_{O2}', 'max \omega_{O2}');
    ylabel('queue length (veh)')
    
    ax(7) = nexttile(7); 
    plot(t_tot(1:step:end), origins_tot.flow(:, 1:step:end)')
    hlegend(7) = legend('q_{O1}', 'q_{O2}');
    ylabel('origin flow (veh/h)')
    
    ax(8) = nexttile(8); hold on,
    ax(8).ColorOrderIndex = 2;
    stairs(t_tot(1:step:end), origins_tot.rate(:, 1:step:end)')
    hlegend(8) = legend('r_{O2}');
    ylabel('metering rate')
    
    linkaxes(ax, 'x')
    for i = 1:length(ax)
        xlabel(ax(i), 'time (h)')
        plot_episodes_separators(ax(i), hlegend(i), ep_tot, Tfin)
        ax(i).YLim(1) = 0;
        ax(i).XLim(2) = t_tot(end);
    end
end

if plot_learning
    % learning quantities figure
    figure;
    tiledlayout(4, 2, 'Padding', 'none', 'TileSpacing', 'compact')
    sgtitle(runname, 'Interpreter', 'none')
    ax = matlab.graphics.axis.Axes.empty;
    
    ax(1) = nexttile(1, [1, 2]);
    performance = arrayfun(@(ep) full(sum(Lrl(origins.queue{ep}, links.density{ep}))), 1:ep_tot);
    performance_only_tts = arrayfun(@(ep) full(sum(TTS(origins.queue{ep}, links.density{ep}))), 1:ep_tot);
    yyaxis left
    plot(linspace(0, ep_tot, length(performance)), performance)
    % stairs(linspace(0, ep_tot, length(performance) + 1), [performance, performance(end)])
    % ax(1).YLim(1) = 0;
    ylabel('J(\pi)')
    yyaxis right
    plot(linspace(0, ep_tot, length(performance_only_tts)), performance_only_tts)
    % stairs(linspace(0, ep_tot, length(performance_only_tts) + 1), [performance_only_tts, performance_only_tts(end)])
    ylabel('TTS(\pi)')
    % bar(0.5:1:(ep_tot-0.5), [performance_only_tts', (performance - performance_only_tts)'], 'stacked')
    % ylabel('J(\pi)')
    
    ax(2) = nexttile(3, [1, 2]);
    td_error_tot = cell2mat(td_error);
    plot(linspace(0, ep_tot, length(td_error_tot)), td_error_tot, 'o', 'MarkerSize', 2)
    ylabel('TD error \tau')
    
    ax_ = nexttile(5);
    L_tot = full(Lrl(origins_tot.queue, links_tot.density));
    plot(t_tot(1:step:end), L_tot(:, 1:step:end))
    plot_episodes_separators(ax_, [], ep_tot, Tfin)
    xlabel('time (h)'), ylabel('L')
    ax_.XLim(2) = t_tot(end);
    
    ax(3) = nexttile(7); hold on
    stairs(linspace(0, ep_tot, length(rl_pars.v_free)), rl_pars.v_free)
    stairs(linspace(0, ep_tot, length(rl_pars.rho_crit)), rl_pars.rho_crit)
    ax(3).ColorOrderIndex = 1;
    plot([0, ep_tot], [true_pars.v_free, true_pars.v_free], '--')
    plot([0, ep_tot], [true_pars.rho_crit, true_pars.rho_crit], '--')
    legend('v_{free}', '\rho_{crit}')
    hold off
    ylabel('v_{free}, \rho_{crit}')
    
    ax(4) = nexttile(6, [2, 1]); hold on
    Markers = {'+','o','*','x','v','d','^','s','>','<'};
    weights = fieldnames(rl_pars);
    legendStrings = {};
    for i = 1:(length(weights) - 2) % first 2 are v_free and rho_crit
        name = weights{i + 2};
        weight = rl_pars.(name);
        for j = 1:size(weight, 1)
            w = (weight(j, :));
            if scaled_learned
                w = rescale(w);
            end
            plot(linspace(0, ep_tot, length(w)), w, 'Marker', Markers{i}, 'MarkerSize', 4)
%             stairs(linspace(0, ep_tot, length(w)), w, 'Marker', Markers{i}, 'MarkerSize', 4)
            legendStrings{end + 1} = append(name, '_', string(j));
        end
    end
    hold off
    legend(legendStrings{:}, 'interpreter', 'none', 'FontSize', 6)
    if scaled_learned
        ylabel('weights (scaled)')
    else
        ylabel('weights')
    end
    
    linkaxes(ax, 'x')
    for i = 1:length(ax)
        xlabel(ax(i), 'episode')
        ax(i).XLim(2) = ep_tot;
    end
end



%% local functions
function plot_episodes_separators(ax, hlegend, episodes, Tfin)
    if episodes <= 1
        return
    end

    if ~isempty(hlegend) && isa(hlegend, 'matlab.graphics.illustration.Legend')
        n_data = length(hlegend.String);
    end

    line(ax, repmat((1:episodes - 1) * Tfin, 2, 1), [0, ax.YLim(2)], ...
    'Color', '#686a70', 'LineStyle', ':', 'LineWidth', 0.75)
%     hold(ax(i), 'on')
%     plot(ax(i), (1:episodes) * Tfin, [0, ax(i).YLim(2)], ':k', 'LineWidth', 0.25)
%     hold(ax(i), 'off')

    if ~isempty(hlegend) && isa(hlegend, 'matlab.graphics.illustration.Legend')
        hlegend.String = hlegend.String(1:n_data);
    end
end
