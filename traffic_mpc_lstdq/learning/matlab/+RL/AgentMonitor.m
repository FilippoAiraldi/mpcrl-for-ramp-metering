classdef AgentMonitor < handle
    % AGENTMONITOR. OpenAI-like gym wrapper for MPC-based RL agents.
    
    properties
        agent (1, 1)                % RL.AgentBase
        %
        mpcruns (1, :) struct       % MPC history
        transitions (1, :) struct   % Transition history
        updates (1, :) struct       % Update history
        % 
        weights (1, :) struct       % history of agent's weight values
    end
    
    properties (Dependent)
        nb_updates (1, 1) double
    end



    methods (Access = public)
        function obj = AgentMonitor(agent)
            % AGENTMONITOR. Construct an empty instance of this class.
            arguments
                agent (1, 1) RL.AgentBase
            end
            obj.agent = agent;
            obj.clear_history();
        end
        
        function [r0_opt, sol, info] = solve_mpc(obj, name, varargin)
            [r0_opt, sol, info] = obj.agent.solve_mpc(name, varargin{:});
            
            % save: type, objval f, success
            s = struct('type', name, 'f', info.f, ...
                       'success', info.success, 'swmax', sol.swmax');
            if isempty(obj.mpcruns)
                obj.mpcruns = s;
            else
                obj.mpcruns(end + 1) = s;
            end
        end

        function s = save_transition(obj, varargin)
            s = obj.agent.save_transition(varargin{:});

            % save: all the returned structure
            if isempty(obj.transitions)
                obj.transitions = s;
            else
                obj.transitions(end + 1) = s;
            end
        end

        function [n, deltas, s] = update(obj, varargin)
            % what to save: descent direction p, Hmod
            [n, deltas, s] = obj.agent.update(varargin{:});

            % save: all the returned structure
            if isempty(obj.updates)
                obj.updates = s;
            else
                obj.updates(end + 1) = s;
            end

            % save also the new weights
            obj.weights(end + 1) = obj.agent.weights.value;
        end

        function clear_history(obj)
            % CLEAR_HISTORY. Clears the agent history saved so far.

            % unlike the traffic monitor, we cannot easily preallocate 
            % nan arrays that will be then filled. In this case, when the
            % length of data to be saved is undefined, either use a cell 
            % array or append to an ever-growing array
            obj.mpcruns = struct.empty;
            obj.transitions = struct.empty;
            obj.updates = struct.empty;
            obj.weights = obj.agent.weights.value;
        end

        function [fig, layout, axs, lgds] = plot_learning(obj, title, ...
                                                          step, logscale)
            % PLOT_LEARNING. PLots the learning-related quantities. A step 
            % size can be provided to reduce the number of datapoints 
            % plotted.
            arguments
                obj (1, 1) RL.AgentMonitor
                title (1, :) char {mustBeTextScalar} = obj.agent.agentname
                step (1, 1) double {mustBePositive, mustBeInteger} = 1
                logscale (1, 1) logical = false
            end

            % create the mpc iterations and updates array
            k_mpc = 0:step:length(obj.transitions) - 1;
            k_up = 1:step:length(obj.updates);

            % create a function to plot on different scales
            if logscale
                do_plot = @(ax, x, y, varargin) ...
                                    semilogy(ax, x, abs(y), varargin{:});
            else
                do_plot = @(ax, x, y, varargin) ...
                                    plot(ax, x, y, varargin{:});
            end

            % prepare data
            td_err = [obj.transitions.td_err];
            td_err_perc = [obj.transitions.td_err_perc];
            g_norm = vecnorm([obj.transitions.g], 2, 1);
            p_norm = vecnorm([obj.updates.p], 2, 1);
            
            % separate RL weights from model parameters
            names = string(fieldnames(obj.weights)');
            rl_mdl_pars = names(~startsWith(names, 'w_'));
            rl_weights = setdiff(names, rl_mdl_pars, 'stable');

            % instantiate figure, layout, axes and legends
            fig = figure('Visible', 'on');
            layout = tiledlayout(fig, 4, 1, 'Padding', 'none', ...
                                 'TileSpacing', 'compact');
            if ~isempty(title)
                sgtitle(layout, title, 'Interpreter', 'none')
            end
            axs = matlab.graphics.axis.Axes.empty;
            lgds = matlab.graphics.illustration.Legend.empty;

            % plot each quantity
            % norms of gradients and search directions
            axs(end + 1) = nexttile;
            clrs = axs(end).ColorOrder(1:2, :);
            do_pobj.lot(axs(end), k_mpc, g_norm, 'Color', clrs(1, :))
            ylabel(axs(end), '||g||');
            xlabel(axs(end), 'transition');
            axs(end).XAxis.Limits = [k_mpc(1), k_mpc(end)];
            axs(end).XColor = clrs(1, :);
            axs(end).YColor = clrs(1, :);
            %
            axs(end + 1) = axes(layout);
            do_plot(axs(end), k_up, p_norm, 'Color', clrs(2, :))
            axs(end).XAxisLocation = 'top';
            axs(end).YAxisLocation = 'right';
            axs(end).Color = 'none';
            axs(end - 1).Box = 'off';
            axs(end).Box = 'off';
            ylabel(axs(end), '||p||');
            xlabel(axs(end), 'update');
            axs(end).XAxis.Limits = [k_up(1), k_up(end)];
            axs(end).XColor = clrs(2, :);
            axs(end).YColor = clrs(2, :);

            % TD errors
            axs(end + 1) = nexttile;
            yyaxis(axs(end), 'left')
            do_plot(axs(end), k_mpc, td_err)
            ylabel(axs(end), 'TD error');
            yyaxis(axs(end), 'right')
            plot(axs(end), k_mpc, td_err_perc)
            ylabel(axs(end), '%');
            xlabel(axs(end), 'transition');
            axs(end).XAxis.Limits = [k_mpc(1), k_mpc(end)];

            % Model parameters learnt via RL
            axs(end + 1) = nexttile;
            hold(axs(end), 'on')
            % plot learned parameters
            for n = rl_mdl_pars
                stairs(axs(end), [0,k_up], [obj.weights.(n)])
            end
            % plot the true parameters
            plot(axs(end), [0, k_up(end)], ...
                obj.agent.env.model.v_free * [1,1], '--k','LineWidth',.5)
            plot(axs(end), [0, k_up(end)], ...
                obj.agent.env.model.rho_crit * [1,1], '--k','LineWidth',.5)
            lgdstrs = [rl_mdl_pars, {'', ''}]; % empty entries for true plots
            hold(axs(end), 'off')
            if logscale
                set(axs(end), 'YScale', 'log');
            end
            ylabel(axs(end), 'Model parameters');
            xlabel(axs(end), 'update')
            axs(end).XAxis.Limits = [0, k_up(end)];
            lgds(end + 1) = legend(lgdstrs{:}, 'interpreter', 'none');

            % RL weights parameters
            markers = {'o', '*', 'x', 'v', 'd', '^', 's', '>', '<', '+'};
            lgdstrs = {}; %#ok<*AGROW> 
            L = size(axs(end).ColorOrder, 1);
            axs(end + 1) = nexttile;
            hold(axs(end), 'on')
            for i = 1:length(rl_weights)
                n = rl_weights(i);
                w = [obj.weights.(n)];
                clr = axs(end).ColorOrder(mod(i - 1, L) + 1, :);
                
                % plot this weight
                for j = 1:size(w, 1)
                    marker = markers{mod(j - 1, length(markers)) + 1};
                    stairs(axs(end), [0, k_up], w(j, :), 'Color', clr, ...
                           'Marker', marker, 'MarkerSize', 4)
                    lgdstrs{end + 1} = ''; 
                end

                % plot an invisible point as legend entry
                plot(nan, nan, '-', 'Color', clr);
                lgdstrs{end + 1} = n; 
            end
            hold(axs(end), 'off')
            if logscale
                set(axs(end), 'YScale', 'log');
            end
            ylabel(axs(end), 'Weights');
            xlabel(axs(end), 'update')
            axs(end).XAxis.Limits = [0, k_up(end)];
            lgds(end + 1) = legend(lgdstrs{:}, 'interpreter', 'none', ...
                                   'FontSize', 6);
            

            % finally show figure
            fig.Visible = 'on';
        end
    end

    % PROPERTY GETTERS
    methods 
        function nb = get.nb_updates(obj)
            nb = length(obj.updates);
        end
    end
end
