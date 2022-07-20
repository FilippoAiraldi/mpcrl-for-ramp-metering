classdef TrafficEnv < handle
    % TRAFFICENV. OpenAI-like gym environment for METANET traffic control
    %  applications.

    properties (GetAccess = public, SetAccess = protected)
        episodes (1, 1) double {mustBePositive, mustBeInteger} = 10
        sim (1, 1) struct
        model (1, 1) struct
        mpc (1, 1) struct
        %
        dynamics (1, 1) struct
        %
        L (1, 1) casadi.Function
        TTS (1, 1) casadi.Function
        RV(1, 1) casadi.Function % rate var
        cumcost (1, 1) struct
        %
        state (1, 1) struct
        demand (:, :) double {mustBeNonnegative}
        %
        r_prev (:, 1) double 
        k_tot (1, 1) double {mustBePositive, mustBeInteger} = 1 % internal total time counter
        is_done (1, 1) logical = false
    end
    properties (Dependent) 
        x (:, 1) double % state but as a vector
        d (:, 1) double % current demand
        d_future (:, :) double % future demands
        %
        ep (1, 1) double {mustBePositive, mustBeInteger} % internal counter of episode
        k (1, 1) double {mustBePositive, mustBeInteger} % internal time counter per episode
    end


    
    methods (Access = public)
        function obj = TrafficEnv(episodes, sim, model, mpc)
            % TRAFFICENV. Constructs an empty instance of this environment.
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
            [obj.L, obj.TTS, obj.RV] = ...
                                METANET.get_stage_cost(sim, model, mpc);
        end

        function [state, errormsg] = reset(obj, r)
            % RESET. Resets the environment to default, random conditions.
            arguments
                obj (1, 1) METANET.TrafficEnv
                r (:, 1) double
            end
            mdl = obj.model;

            % create random demands for all episodes + 1 to avoid 
            % out-of-bound access
            obj.demand = METANET.get_demands( ...
                            obj.sim.t, obj.episodes + 1, mdl.demand_type);
            assert(size(obj.demand, 1) == mdl.n_dist)

            % compute initial state (at steady-state)
            [w, rho, v, ~, ~, errormsg] = METANET.steady_state( ...
                                        obj.dynamics.f, ...
                                        zeros(mdl.n_origins, 1), ...
                                        10 * ones(mdl.n_links, 1), ...
                                        100 * ones(mdl.n_links, 1), ...
                                        r, obj.demand(:, 1), ...
                                        mdl.rho_crit, mdl.a, mdl.v_free);
            state = struct('w', w, 'rho', rho, 'v', v);
            obj.state = state;

            % reset time counter
            obj.k_tot = 1;
            obj.r_prev = r;

            % reset other stats
            obj.is_done = false;
            obj.reset_cumcost()
        end

        function reset_cumcost(obj)
            arguments
                obj (1, 1) METANET.TrafficEnv
            end
            % RESET_CUMCOST. Resets cumulative costs - one for each term:
            %   - J is the total cumulative cost
            %   - TTS is the cumulative cost term due to TTS
            %   - RV is the cumulative cost term due to rate variability
            %   - CV is the cumulative cost term due to constraint violation
            obj.cumcost = struct('J', 0, 'TTS', 0, 'RV', 0, 'CV', 0);
        end

        function [state, cost, done, info] = step(obj, r)
            % STEP. Steps the discrete dynamics by one time step, given a 
            % control ramp flow.
            arguments
                obj (1, 1) METANET.TrafficEnv
                r (:, 1) double % numerical or symbolical
            end
            assert(size(r, 1) == obj.model.n_ramps && ~obj.is_done, ...
                  'Invalid input shape, or environment is done.')
            
            % compute cost of being in the current state
            [cost, TTS_term, RV_term, CV_term] = obj.L( ...
                obj.state.w, obj.state.rho, obj.state.v, r, obj.r_prev);
            cost = full(cost); % a.k.a., L
            TTS_term = full(TTS_term);
            RV_term = full(RV_term);
            CV_term = full(CV_term);

            % step the dynamics (according to the true parameters)
            [q_o, w_next, q, rho_next, v_next] = obj.dynamics.f(...
                        obj.state.w, obj.state.rho, obj.state.v, ... 
                        r, obj.demand(:, obj.k_tot), ...
                        obj.model.rho_crit, obj.model.a, obj.model.v_free);

            % save as state
            state = struct('w', full(w_next), 'rho', full(rho_next), ...
                'v', full(v_next));

            % create info
            info = struct('ep_done', mod(obj.k_tot, obj.sim.K) == 0, ...
                          'q', full(q), 'q_o', full(q_o), ...
                          'TTS', TTS_term, 'RV', RV_term, 'CV', CV_term);

            % increment cost cumulative totals
            obj.cumcost.J = obj.cumcost.J + cost;
            obj.cumcost.TTS = obj.cumcost.TTS + TTS_term;
            obj.cumcost.RV = obj.cumcost.RV + RV_term;
            obj.cumcost.CV = obj.cumcost.CV + CV_term;

            % is the episode over?
            done = obj.k_tot == obj.sim.K * obj.episodes;
            obj.is_done = obj.is_done | done;
            
            % increment/save quantities for next iteration
            obj.k_tot = obj.k_tot + 1;
            obj.r_prev = r;
            obj.state = state;
        end

        function sync(obj, other)
            % SYNC. Syncs the current env with the other's state and 
            % demand, as well as internal variables.
            arguments
                obj (1, 1) METANET.TrafficEnv
                other (1, 1) METANET.TrafficEnv
            end
            obj.demand = other.demand;
            obj.state = other.state;
            obj.k_tot = other.k_tot;
            obj.r_prev = other.r_prev;
            obj.is_done = other.is_done;
            obj.cumcost = other.cumcost;
        end
    end

    % PROPERTY GETTERS
    methods
        function x = get.x(obj)
            % X. Gets the current state as a vector.
            x = cell2mat(struct2cell(obj.state));
        end

        function d = get.d(obj)
            % D. Gets the current demand as a vector.
            d = obj.demand(:, obj.k_tot);
        end

        function d = get.d_future(obj)
            % D. Gets the future demands (starting from the current 
            % timestep) as a matrix.
            k_fin = obj.k_tot + obj.mpc.M * obj.mpc.Np - 1;
            d = obj.demand(:, obj.k_tot:k_fin);
        end

        function k = get.k(obj)
            % K. Gets the current time index for the episode.
            k = mod(obj.k_tot - 1, obj.sim.K) + 1;
        end

        function ep = get.ep(obj)
            % EP. Gets the current episode index.
            ep = ceil(obj.k_tot / obj.sim.K);
        end
    end
end
