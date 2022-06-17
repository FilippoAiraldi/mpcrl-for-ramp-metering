classdef MonitorWrapper < handle
    % MONITORWRAPPER. OpenAI-like gym environment wrapper for the recording
    % and plotting of traffic quantities.

    properties (GetAccess = public, SetAccess = protected)
        env (1, 1) % METANET.TrafficEnv
        %
        iterations (1, 1) double {mustBePositive, mustBeInteger} = 10
        iter (1, 1) double {mustBePositive, mustBeInteger} = 1 % internal iteration counter 
        % 
        origins (1, 1) struct
        links (1, 1) struct
        exec_times (:, :) double
    end

    properties (Access = private)
        last_ep_start (1, 1) uint64 = 0
    end
    


    methods (Access = public)
        function obj = MonitorWrapper(env, iterations)
            % MONITORWRAPPER. Constructs an instance of this environment.
            arguments
                env (1, 1) METANET.TrafficEnv
                iterations (1, 1) double {mustBePositive, mustBeInteger}
            end
            obj.env = env;
            obj.iterations = iterations;
            obj.clear_history();
        end

        function state = reset(obj, varargin)
            % just class the wrapped env's reset function
            state = obj.env.reset(varargin{:});
        end

        function [next_state, cost, done, info] = step(obj, r)
            % Since the dynamics return the next state (k+1) but also some 
            % current quantities (k), first save the state (k), then step 
            % (k->k+1), and then save the other quantities (k).
            i = obj.iter;
            e = obj.env.ep;
            k = obj.env.k;

            % check if current episode's has been started
            if obj.last_ep_start == 0
                obj.last_ep_start = tic;
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

            % reset the structures
            obj.origins = struct; 
            obj.origins.queue = nan(I, E, n_origins, K);
            obj.origins.flow = nan(I, E, n_origins, K);
            obj.origins.demand = nan(I, E, n_dist, K);
            obj.links = struct;
            obj.links.flow = nan(I, E, n_links, K);
            obj.links.density = nan(I, E, n_links, K);
            obj.links.speed = nan(I, E, n_links, K);

            % reset execution time array
            obj.exec_times = nan(I, E);
        end

        function s = state(obj, k)
            % STATE. Returns the state of the system at time step k (k 
            % behaves like a Python index).
            arguments
                obj (1, 1) METANET.MonitorWrapper
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

        function [fig, layout, axs, lgds] = plot(obj, title, step)
            % PLOT. PLots the origins and links traffic-related quantities. 
            % A step size can be provided to reduce the number of 
            % datapoints plotted.
            arguments
                obj (1, 1) METANET.MonitorWrapper
                title (1, :) char {mustBeTextScalar} = char.empty
                step (1, 1) double {mustBePositive, mustBeInteger} = 3
            end
            I = obj.iterations;     % number of learning iterations
            E = obj.env.episodes;   % number of episodes per iteration
            K = obj.env.sim.K;      % number of timesteps per episode

            % create the time array
            t = (0:step:(I * E * K - 1)) * obj.env.sim.T;

            % convert iteration-episode cells to one big array
            origins_ = util.reshape4d(obj.origins);
            links_ = util.reshape4d(obj.links);

            % instantiate figure, layout, axes and legends
            fig = figure;
            layout = tiledlayout(fig, 4, 2, 'Padding', 'none', ...
                                       'TileSpacing', 'compact');
            if ~isempty(title)
                sgtitle(layout, title, 'Interpreter', 'none')
            end
            axs = matlab.graphics.axis.Axes.empty;
            lgds = matlab.graphics.illustration.Legend.empty;

            % plot each quantity
            items = { ...
                'speed (km/h)', ...
                        {'v_{L1}', 'v_{L2}', 'v_{L3}'}, links_.speed; ...
                'flow (veh/h)', ...
                        {'q_{L1}', 'q_{L2}', 'q_{L3}'}, links_.flow; ...
                'density (veh/km/lane)', ...
                        {'\rho_{L1}', '\rho_{L2}', '\rho_{L3}'}, ...
                                links_.density; ...
                'origin 1 flow (veh/h)', ...
                        {'q_{O1}'}, origins_.flow(1, :); ...
                'origin demand (veh/h)', ...
                        {'d_{O1}', 'd_{O2}'}, origins_.demand(1:2, :); ...
                'origin 2 flow (veh/h)', {'q_{O2}'}, origins_.flow(2, :);
                'boundary downstream\newlinecongestion (veh/km/lane)', ...
                        {'d_{cong}'}, ...
                                    origins_.demand(3, :); ...
                'queue length (veh)', ...
                        {'\omega_{O1}', '\omega_{O2}'}, origins_.queue; ...
                
            };
            for i = 1:size(items, 1)
                ylbl = items{i, 1};
                names = items{i, 2};
                data = items{i, 3};
                axs(i) = nexttile(i);
                plot(axs(i), t, data(:, 1:step:end)');
                lgds(i) = legend(axs(i), names{:});
                ylabel(axs(i), ylbl);
            end
            
            % further customizations
            % change color of 7 to second default color
            axs(6).Children.Color = axs(6).ColorOrder(2, :);

            % change color of 7 to third default color
            axs(7).Children.Color = axs(7).ColorOrder(3, :);

            % add max queue to 8
            max_queue = obj.env.model.max_queue;
            hold(axs(8), 'on');
            plot(axs(8), [t(1), t(end)], [1, 1] * max_queue, '-.k')
            hold(axs(8), 'off');
            lgds(8).String{end} = 'max \omega';
        end
    end
end
