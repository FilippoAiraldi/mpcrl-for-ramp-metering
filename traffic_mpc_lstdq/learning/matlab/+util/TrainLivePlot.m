classdef TrainLivePlot < handle
    % TRAINLIVEPLOT. Class for plotting live how the training is going.
    
    properties (GetAccess = public, SetAccess = protected)
        env (1, 1)          % METANET.TrafficMonitor
        agent (1, 1)        % RL.AgentMonitor
    end

    properties (Access = private)
        fig                 % handle to Figure
        plot                % handle to function for plotting
        plots (1, 1) struct % handle to plots
        last_tr (1, 1) double
        last_up (1, 1) double
    end



    methods (Access = public)
        function obj = TrainLivePlot(env, agent, logscale)
            % TRAINLIVEPLOT. Instantiates the class.
            arguments
                env (1, 1) METANET.TrafficMonitor
                agent (1, 1) RL.AgentMonitor
                logscale (1, 1) logical = true
            end
            obj.env = env;
            obj.agent = agent;

            obj.fig = [];
            if logscale
                obj.plot = @(ax, y, varargin) semilogy(ax, abs(y), ...
                                                            varargin{:});
            else
                obj.plot = @(ax, y, varargin) plot(ax, y, varargin{:});
            end
        end

        function delete(obj)
            try
                close(obj.fig)
            end
        end

        function plot_episode_summary(obj, iter, ep)
            % PLOT_EP_SUMMARY. Plots a recap of the episode just finished 
            % (assumes the agent has not been yet reset).
            arguments
                obj (1, 1) util.TrainLivePlot
                iter (1, 1) double {mustBeInteger, mustBePositive}
                ep (1, 1) double {mustBeInteger, mustBePositive}
            end
            if isempty(obj.fig) || ~isvalid(obj.fig)
                obj.create_plots(iter, ep);
            else
                obj.add_to_plots(iter, ep);
            end          
        end
    end

    methods (Access = private)
        function [J, TTS, g_norm, p_norm, CV] = get_data_to_plot( ...
                                        obj, it1, it2, ep1, ep2, tr1, up1)
            e = obj.env;
            ee = e.env;
            a = obj.agent;

            J = util.flatten(sum(e.cost.L(it1:it2, ep1:ep2, :), 3));
            TTS = util.flatten(sum(e.cost.TTS(it1:it2, ep1:ep2, :), 3));
            CV = util.flatten( ...
                    sum(e.origins.queue(it1:it2, ep1:ep2, 2, :) > ...
                            ee.model.max_queue, 4)) / ee.sim.K * 100;

            % for these two, pay attention if the structs are empty
            if isempty(a.transitions)
                g_norm = nan;
            else
                g_norm = mean(vecnorm([a.transitions(tr1:end).g], 2, 1));
            end
            if isempty(a.updates)
                p_norm = nan;
            else
                p_norm = vecnorm([a.updates(up1:end).p], 2, 1);
            end
        end

        function create_plots(obj, iter, ep)
            [J, TTS, g_norm, p_norm, CV] = obj.get_data_to_plot( ...
                                                    1, iter, 1, ep, 1, 1);
            a = obj.agent;

            % save the last index that has been plotting
            obj.last_tr = length(a.transitions);
            obj.last_up = length(a.updates);
    
            % create figure
            obj.fig = figure('Visible', 'on');
            layout = tiledlayout(obj.fig, 4, 1, 'Padding', 'none', ...
                        'TileSpacing', 'compact');
            if ~isempty(a.agent.name)
                sgtitle(layout, a.agent.name, 'Interpreter', 'none')
            end
            
            % structure containing plots
            obj.plots = struct;

            % cost terms
            ax = nexttile(layout); 
            yyaxis(ax, 'left'), 
            obj.plots.J = obj.plot(ax, J);
            ylabel('J')
            yyaxis(ax, 'right'), 
            obj.plots.TTS = obj.plot(ax, TTS);
            ylabel(ax, 'TTS(\pi)');
            xlabel(ax, 'episode');

            % g norm
            ax = nexttile(layout); 
            obj.plots.g_norm = obj.plot(ax, g_norm);
            ylabel(ax, '||g|| (averaged)');
            xlabel(ax, 'episode');

            % p norm
            ax = nexttile(layout); 
            obj.plots.p_norm = obj.plot(ax, p_norm);
            ylabel(ax, '||p||');
            xlabel(ax, 'update');

            % constraint violation
            ax = nexttile(layout); 
            obj.plots.CV = area(ax, CV);
            ylabel(ax, 'Constr. violation');
            xlabel(ax, 'episode');
            ytickformat(ax, 'percentage');
        end

        function add_to_plots(obj, iter, ep)
            [J, TTS, g_norm, p_norm, CV] = obj.get_data_to_plot( ...
                                    iter, iter, ep, ep, ...
                                    obj.last_tr + 1, obj.last_up + 1);
            
            % update plots
            obj.plots.J.YData = [obj.plots.J.YData, J];
            obj.plots.TTS.YData = [obj.plots.TTS.YData, TTS];
            obj.plots.CV.YData = [obj.plots.CV.YData, CV];
            if isnan(obj.plots.g_norm.YData)
                obj.plots.g_norm.YData = g_norm;
            else
                obj.plots.g_norm.YData = [obj.plots.g_norm.YData, g_norm];
            end
            if isnan(obj.plots.p_norm.YData)
                obj.plots.p_norm.YData = p_norm;
            else
                obj.plots.p_norm.YData = [obj.plots.p_norm.YData, p_norm];
            end

            % update last used indices
            obj.last_tr = length(obj.agent.transitions);
            obj.last_up = length(obj.agent.updates);
        end
    end
end
