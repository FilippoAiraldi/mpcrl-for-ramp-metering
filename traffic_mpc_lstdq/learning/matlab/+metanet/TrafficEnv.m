classdef TrafficEnv < handle
    % TRAFFICENV. OpenAI-like gym environment for METANET traffic control
    %  applications.

    properties (GetAccess = public, SetAccess = protected)
        episodes (1, 1) double {mustBePositive, mustBeInteger} = 50
        sim (1, 1) struct
        model (1, 1) struct
        mpc (1, 1) struct
        %
        dynamics (1, 1) struct
        L (1, 1) casadi.Function
        TTS (1, 1) casadi.Function
        Rate_Var(1, 1) casadi.Function
        %
        state (1, 1) struct
        demand (:, :) double {mustBeNonnegative}
        %
        r_prev (:, 1) double 
        %
        k_tot (1, 1) double {mustBePositive,mustBeInteger} = 1 % internal total time counter
    end
    % history.origins.queue{iteration}{episode}(time step in the episode)

    properties (Dependent) 
        x (:, 1) double % state but as a vector
        d (:, 1) double % current demand
        %
        ep (1, 1) double {mustBePositive,mustBeInteger} % internal counter of episode
        k (1, 1) double {mustBePositive,mustBeInteger} % internal time counter per episode
    end


    
    methods (Access = public)
        function obj = TrafficEnv(episodes, sim, model, mpc)
            % TRAFFICENV. Constructs an empty instance of this gym.
            arguments
                episodes (1, 1) double {mustBePositive, mustBeInteger}
                sim (1, 1) struct
                model (1, 1) struct
                mpc (1, 1) struct % only needed to build stage-cost
            end
            % save to instance
            obj.episodes = episodes;
            obj.sim = sim;
            obj.model = model;
            obj.mpc = mpc;

            % create dynamics function 
            obj.dynamics = METANET.get_dynamics(sim, model);

            % create stage-cost function
            [obj.L, obj.TTS, obj.Rate_Var] = ...
                                METANET.get_stage_cost(sim, model, mpc);
        end

        function reset(obj, r)
            % RESET. Resets the environment to default, random conditions.
            arguments
                obj (1, 1) METANET.TrafficEnv
                r (:, 1) double = obj.model.C(2)
            end
            mdl = obj.model;

            % create random demands for all episodes + 1 to avoid 
            % out-of-bound access
            obj.demand = METANET.get_demands( ...
                                    obj.sim.t, obj.episodes + 1, 'fixed');
            assert(size(obj.demand, 1) == mdl.n_dist)

            % compute initial state (at steady-state)
            [w, rho, v] = METANET.steady_state(obj.dynamics.f, ...
                                       zeros(mdl.n_origins, 1), ...
                                       10 * ones(mdl.n_links, 1), ...
                                       100 * ones(mdl.n_links, 1), ...
                                       r, obj.demand(:, 1), ...
                                       mdl.rho_crit,mdl.a,mdl.v_free);
            obj.state = struct('w', w, 'rho', rho, 'v', v);

            % reset time counter
            obj.k_tot = 1;
            obj.r_prev = r;
        end

        function [state, cost, done, info] = step(obj, r)
            % STEP. Steps the discrete dynamics by one time step, given a 
            % control ramp flow.
            arguments
                obj (1, 1) METANET.TrafficEnv
                r (:, 1) double % numerical or symbolical
            end
            assert(size(r, 1) == obj.model.n_ramps)
            
            % compute cost of being in the current state
            cost = full(obj.L(obj.state.w, obj.state.rho, obj.state.v, ...
                              r, obj.r_prev));

            % step the dynamics (according to the true parameters)
            [q_o, w_next, q, rho_next, v_next] = obj.dynamics.f(...
                        obj.state.w, obj.state.rho, obj.state.v, ... 
                        r, obj.demand(:, obj.k_tot), ...
                        obj.model.rho_crit, obj.model.a, obj.model.v_free);

            % save as state
            state = struct('w', full(w_next), 'rho', full(rho_next), ...
                'v', full(v_next));
            info = struct('q', full(q), 'q_o', full(q_o));

            % is the episode over?
            done = obj.k_tot == obj.sim.K * obj.episodes; % or the previous iter?
            
            % save quantities for next iteration
            obj.k_tot = obj.k_tot + 1;
            obj.r_prev = r;
            obj.state = state;
        end
    end


    % PROPERTY GETTERS
    methods
        function x = get.x(obj)
            x = cell2mat(struct2cell(obj.state));
        end

        function d = get.d(obj)
            d = obj.demand(:, obj.k_tot);
        end

        function k = get.k(obj)
            k = mod(obj.k_tot - 1, obj.sim.K) + 1;
        end

        function ep = get.ep(obj)
            ep = ceil(obj.k_tot / obj.sim.K);
        end
    end
end
