classdef TrafficMonitor < handle
    % TRAFFICMONITOR. OpenAI-like gym environment wrapper for the recording
    % and plotting of traffic quantities.

    properties (GetAccess = public, SetAccess = protected)
        env (1, 1) % METANET.TrafficEnv
        %
        iterations (1, 1) double {mustBePositive, mustBeInteger} = 10
        iter (1, 1) double {mustBePositive, mustBeInteger} = 1 % internal iteration counter 
        % 
        origins (1, 1) struct
        links (1, 1) struct
        cost (1, 1) struct
        exec_times (:, :) double
    end

    properties (Access = private)
        last_ep_start (1, 1) uint64 = tic
        no_tic (1, 1) logical = true
    end

    properties (Dependent)
        current_ep_time (1, 1) double
    end
    


    methods (Access = public)
        function obj = TrafficMonitor(env, iterations)
            % TRAFFICMONITOR. Constructs an instance of this environment.
            arguments
                env (1, 1) METANET.TrafficEnv
                iterations (1, 1) double {mustBePositive, mustBeInteger}
            end
            obj.env = env;
            obj.iterations = iterations;
            obj.clear_history();
        end

        function varargout = reset(obj, varargin)
            [varargout{1:nargout}] = obj.env.reset(varargin{:});
        end

        function [next_state, cost, done, info] = step(obj, r)
            % Since the dynamics return the next state (k+1) but also some 
            % current quantities (k), first save the state (k), then step 
            % (k->k+1), and then save the other quantities (k).
            i = obj.iter;
            e = obj.env.ep;
            k = obj.env.k;

            % if time has not already been initialized, do it now
            if obj.no_tic
                obj.last_ep_start = tic;
                obj.no_tic = false;
            end

            % save current state and demand
            obj.origins.queue(i, e, :, k) = obj.env.state.w;
            obj.links.density(i, e, :, k) = obj.env.state.rho;
            obj.links.speed(i, e, :, k) = obj.env.state.v;
            obj.origins.demand(i, e, :, k) = obj.env.d;

            % step the system
            [next_state, cost, done, info] = obj.env.step(r);

            % save current flows 
            obj.origins.flow(i, e, :, k) = info.q_o; % (a.k.a., control action r itself)
            obj.links.flow(i, e, :, k) = info.q;
            
            % save costs
            obj.cost.L(i, e, k) = cost;
            obj.cost.TTS(i, e, k) = info.TTS;
            obj.cost.RV(i, e, k) = info.RV;
            obj.cost.CV(i, e, k) = info.CV;

            % increment iteration counter if all the episodes are done
            obj.iter = obj.iter + done; % done is logical

            % if current episode is done, save its execution time
            if info.ep_done
                obj.exec_times(i, e) = toc(obj.last_ep_start);
                obj.last_ep_start = tic;
            end
        end

        function clear_history(obj)
            % CLEAR_HISTORY. Clears the traffic history saved so far.
            I = obj.iterations;     % number of learning iterations
            E = obj.env.episodes;   % number of episodes per iteration
            K = obj.env.sim.K;      % number of timesteps per episode
            n_links = obj.env.model.n_links;
            n_origins = obj.env.model.n_origins;
            n_dist = obj.env.model.n_dist;

            % reset traffic structures
            obj.origins = struct; 
            obj.origins.queue = nan(I, E, n_origins, K);
            obj.origins.flow = nan(I, E, n_origins, K);
            obj.origins.demand = nan(I, E, n_dist, K);
            obj.links = struct;
            obj.links.flow = nan(I, E, n_links, K);
            obj.links.density = nan(I, E, n_links, K);
            obj.links.speed = nan(I, E, n_links, K);
            
            % reset cost structure
            obj.cost = struct;
            obj.cost.L = nan(I, E, K);
            obj.cost.TTS = nan(I, E, K); 
            obj.cost.RV = nan(I, E, K);
            obj.cost.CV = nan(I, E, K);

            % reset execution time array
            obj.exec_times = nan(I, E);
        end

        function s = state(obj, k)
            % STATE. Returns the state of the system at time step k (k 
            % behaves like a Python index).
            arguments
                obj (1, 1) METANET.TrafficMonitor
                k (1, 1) double {mustBeInteger} = 0
            end
            k_ = obj.env.k;
            if k == 0 || k == k_
                s = obj.env.state;
            else
                % If positive, pick the state from the start of the current
                % episode. If negative, pick it backwards from the current 
                % time in the current episode
                i = obj.iter;
                e = obj.env.ep;
                if k < 0
                    k = k_ + k;
                end
                s = struct( ...
                        'w', squeeze(obj.origins.queue(i, e, :, k)), ... 
                        'rho', squeeze(obj.links.density(i, e, :, k)), ...
                        'v', squeeze(obj.links.speed(i, e, :, k)));
            end
        end

        function s = x(obj, k)
            % X. Returns the state as a vector of the system at time step k
            % (k behaves like a Python index).
            if nargin < 2
                k = 0;
            end
            s = cell2mat(struct2cell(obj.state(k)));
        end

        function [fig, layout, axs, lgds] = plot_traffic(obj, title, step)
            % PLOT_TRAFFIC. PLots the origins and links traffic-related 
            % quantities. A step size can be provided to reduce the number 
            % of datapoints plotted.
            arguments
                obj (1, :) METANET.TrafficMonitor
                title (1, :) char {mustBeTextScalar} = char.empty
                step (1, 1) double {mustBePositive, mustBeInteger} = 3
            end
            % if multiple envs are passed, it is assumed they share the 
            % same constants.   
            Ne = length(obj);          % number of envs to plot on same figure
            assert(Ne < 4, 'cannot handle more than 3 envs in same plot.')
            I = obj(1).iterations;     % number of learning iterations
            E = obj(1).env.episodes;   % number of episodes per iteration
            K = obj(1).env.sim.K;      % number of timesteps per episode

            % create the time array
            t = (0:step:(I * E * K - 1)) * obj(1).env.sim.T;

            % utility function
            afun = @(f, x) arrayfun(f, x, 'UniformOutput', false);

            % flatten arrays
            origins_ = arrayfun(@(o) util.flatten(o.origins), obj);
            links_ = arrayfun(@(o) util.flatten(o.links), obj);

            % instantiate figure, layout, axes and legends
            fig = figure('Visible', 'off');
            layout = tiledlayout(fig, 4, 2, 'Padding', 'none', ...
                                       'TileSpacing', 'compact');
            if ~isempty(title)
                sgtitle(layout, title, 'Interpreter', 'none')
            end
            axs = matlab.graphics.axis.Axes.empty;
            lgds = matlab.graphics.illustration.Legend.empty;
            fmts = { 
                'LineStyle', '-',  'LineWidth', 0.5;
                'LineStyle', '--', 'LineWidth', 0.5;
                'LineStyle', ':',  'LineWidth', 1;
            };

            % plot each quantity
            items = { ...
                'speed (km/h)', ...
                        {'v_{L1}', 'v_{L2}', 'v_{L3}'}, {links_.speed};
                'flow (veh/h)', ...
                        {'q_{L1}', 'q_{L2}', 'q_{L3}'}, {links_.flow};
                'density (veh/km/lane)', ...
                        {'\rho_{L1}', '\rho_{L2}', '\rho_{L3}'}, ...
                                {links_.density}; ...
                'origin 1 flow (veh/h)', ...
                        {'q_{O1}'}, afun(@(o) o.flow(1, :), origins_);
                'origin demand (veh/h)', ...
                        {'d_{O1}', 'd_{O2}'}, ...
                            afun(@(o) o.demand(1:2, :), origins_);
                'origin 2 flow (veh/h)', {'q_{O2}'}, ...
                            afun(@(o) o.flow(2, :), origins_);
                'boundary downstream\newlinecongestion (veh/km/lane)', ...
                        {'d_{cong}'}, ...
                            afun(@(o) o.demand(3, :), origins_);
                'queue length (veh)', ...
                        {'\omega_{O1}', '\omega_{O2}'}, {origins_.queue}; 
                
            };
            for i = 1:size(items, 1)
                ylbl = items{i, 1};
                names = items{i, 2};
                data = items{i, 3};
                axs(i) = nexttile(i);
                hold(axs(i), 'on');
                for e = 1:Ne
                    axs(i).ColorOrderIndex = 1;
                    plot(axs(i), t, data{e}(:, 1:step:end)', fmts{e, :});
                end
                hold(axs(i), 'off');
                lgds(i) = legend(axs(i), names{:});
                xlabel(axs(i), 'time (h)')
                ylabel(axs(i), ylbl);
                axs(i).XAxis.Limits = [t(1), t(end)];
            end
            linkaxes(axs, 'x')
            
            % further customizations
            % change color of 7 to second default color
            for e = 1:Ne
                axs(6).Children(e).Color = axs(6).ColorOrder(2, :);            
    
                % change color of 7 to third default color
                axs(7).Children(e).Color = axs(7).ColorOrder(3, :);
            end

            % add max queue to 8
            max_queue = obj(1).env.model.max_queue;
            hold(axs(8), 'on');
            plot(axs(8), [t(1), t(end)], [1, 1] * max_queue, '-.k')
            hold(axs(8), 'off');
            lgds(8).String{end} = 'max \omega';

            % finally show figure
            fig.Visible = 'on';
        end

        function [fig, layout, axs] = plot_cost(obj, title, step)
            % PLOT_COST. PLots traffic costs. A step size can be provided 
            % to reduce the number of datapoints plotted.
            arguments
                obj (1, :) METANET.TrafficMonitor
                title (1, :) char {mustBeTextScalar} = char.empty
                step (1, 1) double {mustBePositive, mustBeInteger} = 1
            end
            % if multiple envs are passed, it is assumed they share the 
            % same constants.   
            Ne = length(obj);          % number of envs to plot on same figure
            assert(Ne < 4, 'cannot handle more than 3 envs in same plot.')
            I = obj(1).iterations;     % number of learning iterations
            E = obj(1).env.episodes;   % number of episodes per iteration
            K = obj(1).env.sim.K;      % number of timesteps per episode

            % create the step array
            k = 0:step:(I * E * K - 1);
            ep = linspace(k(1), k(end), I * E); % no stepping here

            % utility functions
            afun = @(f, x) arrayfun(f, x, 'UniformOutput', false);
            cfun = @(f, x) cellfun(f, x, 'UniformOutput', false);

            % compute the episode-average of total cost 
            J = afun(@(o) util.flatten(sum(o.cost.L, 3)), obj); 

            % instantiate figure, layout, axes and legends
            fig = figure('Visible', 'off');
            layout = tiledlayout(fig, 4, 2, 'Padding', 'none', ...
                                       'TileSpacing', 'compact');
            if ~isempty(title)
                sgtitle(layout, title, 'Interpreter', 'none')
            end
            axs = matlab.graphics.axis.Axes.empty;
            fmts = { 
                'LineStyle', '-',  'LineWidth', 0.5;
                'LineStyle', '--', 'LineWidth', 0.5;
                'LineStyle', ':',  'LineWidth', 1;
            };

            % plot each quantity
            items = { ...
                'L', afun(@(o) o.cost.L, obj); ...
                'TTS', afun(@(o) o.cost.TTS, obj); ...
                'Rate Variability', afun(@(o) o.cost.RV, obj); ...
                'Constr. Violation', afun(@(o) o.cost.CV, obj); ...
            };
            for i = 1:size(items, 1)
                ylbl = items{i, 1};
                data = items{i, 2};
                
                % plot continuous cost
                data_cont = cfun(@(d) util.flatten(d), data);
                axs(i, 1) = nexttile(2 * i - 1);
                clr = axs(i, 1).ColorOrder(i, :);
                hold(axs(i, 1), 'on');
                for e = 1:Ne
                    plot(axs(i, 1), k, data_cont{e}(:, 1:step:end)', ...
                         'Color', clr, fmts{e, :});
                end
                hold(axs(i, 1), 'off');
                xlabel(axs(i, 1), 'step')
                ylabel(axs(i, 1), ylbl);
                axs(i, 1).XAxis.Limits = [k(1), k(end)];
                
                % plot episode-average cost
                data_avg = cfun(@(d) util.flatten(sum(d, 3)), data);
                axs(i, 2) = nexttile(2 * i);
                hold(axs(i, 2), 'on');
                for e = 1:Ne
                    plot(axs(i, 2), ep, data_avg{e}, 'Color', clr, ...
                        'Marker', 'o', fmts{e, :});
                end
                hold(axs(i, 2), 'off');
                ylabel(axs(i, 2), ylbl);
                
                % as percentage of the total cost as well
                if i > 1
                    yyaxis(axs(i, 2), 'right')
                    clra = [clr, 0.5];
                    hold(axs(i, 2), 'on');
                    for e = 1:Ne
                        plot(axs(i, 2), ep, data_avg{e} ./ J{e} * 100, ...
                            'Color', clra, 'Marker', '^', ...
                            'Markersize', 3, 'MarkerFaceColor', clr, ...
                            fmts{e, :});
                    end
                    hold(axs(i, 2), 'off');
                    axs(i, 2).YAxis(2).Color = axs(i, 2).YAxis(1).Color;
                    ylabel(axs(i, 2), '%');
                end
                
                % for both axis
                step_xticks = floor(length(ep) / 4);
                axs(i, 2).XTick = ep(1:step_xticks:end);
                axs(i, 2).XTickLabel = 0:step_xticks:(I * E - 1);
                xlabel(axs(i, 2), 'episode')
                axs(i, 2).XAxis.Limits = [k(1), k(end)];
            end
            linkaxes(axs, 'x')

            % finally show figure
            fig.Visible = 'on';
        end
    end

    % PROPERTY GETTERS
    methods 
        function t = get.current_ep_time(obj)
            t = toc(obj.last_ep_start);
        end
    end
end