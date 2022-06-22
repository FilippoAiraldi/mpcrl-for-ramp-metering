classdef (Abstract) AgentBase < handle
    % AGENTBASE. An abstract base class for RL agents for traffic control 
    % with an MPC scheme as function approximator. 
    
    properties (GetAccess = public, SetAccess = protected)
        env (1, 1) % METANET.TrafficEnv 
        Q (1, 1) % MPC.NMPC
        V (1, 1) % MPC.NMPC
        %
        weights (1, 1) struct
        % 
        last_sol = [] % (1, 1) struct
        last_info = [] % (1, 1) struct
    end

    properties (Access = public)
        Qopts (1, 1) struct % Q specific options for running Q MPC
        Vopts (1, 1) struct % V specific options for running V MPC
    end

    properties (Dependent)
        w (:, 1) double
        weightnames (1, :) string
    end

    

    methods (Access = public)
        function obj = AgentBase(env, known_mdl_pars)
            arguments
                env (1, 1) METANET.TrafficEnv
                known_mdl_pars (1, 1) struct
            end
            
            % create the NMPC instances for Q(s,a) and V(s)
            obj.env = env;
            [obj.Q, obj.V] = MPC.get_mpcs(env.sim, env.model, env.mpc, ...
                                          env.dynamics, env.TTS, env.RV);
            obj.Qopts = struct('shiftvals', false); % because Q(s,a) is run just after V(s+)
            obj.Vopts = struct('shiftvals', true);

            % initialize solvers
            obj.Q.init_solver(env.mpc.opts_ipopt);
            obj.V.init_solver(env.mpc.opts_ipopt);

            % initialize learnable parameters/weights
            obj.init_pars(known_mdl_pars); 
        end
    
        function [r0_opt, sol, info] = solve_mpc( ...
                            obj, name, pars, state, demand, rlpars, sol0)
            % SOLVE_MPC. Computes the value function V(s) or Q(s,a).
            arguments
                obj (1, 1) RL.AgentBase
                name {mustBeMember(name, {'Q', 'V'})}
                pars (1, 1) struct
                state = [] % (1, 1) struct
                demand double = [] % (n_dist, M * Np)
                rlpars = [] % (1, 1) struct
                sol0 = [] % (1, 1) struct
            end
            env_ = obj.env;

            % if not provided, use the latest agent's weights
            if isempty(rlpars)
                rlpars = obj.weights.value;
            end

            % if the state which to solve the MPC for is not provided, use 
            % current
            if isempty(state)
                s0 = env_.state;
            else
                s0 = state;
            end

            % if demands are not provided, use the future demands based on
            % the current timestep of the environment
            if isempty(demand)
                demand = env_.d_future;
            end

            % if provided, use sol0 to warmstart the MPC. If not provided,
            % use the last_sol field. If the latter is not available yet, 
            % just use some default values
            if isempty(sol0)
                if ~isempty(obj.last_sol)
                    sol0 = obj.last_sol;
                else
                    M = env_.mpc.M;
                    Np = env_.mpc.Np;
                    Nc = env_.mpc.Nc;
                    sol0 = struct( ...
                        'w', repmat(s0.w, 1, M * Np + 1), ...
                        'rho', repmat(s0.rho, 1, M * Np + 1), ...
                        'v', repmat(s0.v, 1, M * Np + 1), ...
                        'r', repmat(env_.r_prev, 1, Nc), ...
                        'swmax', zeros(size(obj.V.vars.swmax)));
                end
            end

            % build the parameters to pass to the MPC
            pars.w0 = s0.w;             % initial state
            pars.rho0 = s0.rho;
            pars.v0 = s0.v;
            pars.d = demand;            % future demands
            for n = obj.weightnames     % rl weights
                pars.(n) = rlpars.(n);
            end

            % call the MPC 
            opts = obj.(strcat(name, 'opts'));
            [sol, info] = obj.(name).solve( ...
                        pars, sol0, opts.shiftvals, env_.mpc.multistart);
            r0_opt = sol.r(:, 1);

            % save last solution
            obj.last_sol = sol;
            obj.last_info = info;
        end

        function varargout = solve_V(obj, varargin)
            % SOLVE_V. Computes the value function V(s). See SOLVE_MPC
            % for more details.
            [varargout{1:nargout}] = obj.solve_mpc('V', varargin{:});
        end

        function varargout = solve_Q(obj, varargin)
            % SOLVE_Q. Computes the value function Q(s,a). See SOLVE_MPC 
            % for more details.
            [varargout{1:nargout}] = obj.solve_mpc('Q', varargin{:});
        end

        function m = rand_perturbation(obj, ep)
            % RAND_PERTURBATION. Returns a random perturbation value based 
            % on the episode (probability of perturbation decreases per 
            % episode). 
            if rand < 0.1 * exp(-(ep - 1) / 5)
                m = obj.env.mpc.perturb_mag * exp(-(ep - 1) / 5) * randn;
            else
                m = 0;
            end
        end
    end

    methods (Abstract)
        save_transition
        update
    end


    % PROPERTY GETTERS
    methods 
        function n = get.weightnames(obj)
            % W. Gets the weight names.
            n = string(fieldnames(obj.weights.bound))';
        end

        function v = get.w(obj)
            % W. Gets the weight values as a vector.
            v = cell2mat(struct2cell(obj.weights.value));
        end

        function v = w_sym(obj, mpcname)
            % W_SYM. Gets the weight values as a vector, but symbolical.
            arguments
                obj (1, 1) RL.AgentBase
                mpcname {mustBeMember(mpcname, {'Q', 'V'})}
            end
            p = struct2cell(obj.weights.sym.(mpcname));
            v = vertcat(p{:});
        end
    end

    methods (Access = private)
        function init_pars(obj, known_pars)
            % compute some sizes
            rho_max = obj.env.model.rho_max;
            sizeV = size(obj.V.pars.w_V);
            sizeL = size(obj.V.pars.w_L);
            sizeT = size(obj.V.pars.w_T);
            sizeS = size(obj.V.pars.w_swmax);
            CV_pen = obj.env.mpc.CV_penalty;
            RV_pen = obj.env.mpc.RV_penalty;
            
            % create struct with all parameters and bounds
            pars = { ...
                'rho_crit', known_pars.rho_crit, [10, rho_max * 0.9]; ...
                'v_free', known_pars.v_free, [30, 300]; ...
                'v_free_track', known_pars.v_free, [30, 300]; ...
                'w_V', ones(sizeV), [-inf, inf]; ...
                'w_L', ones(sizeL), [0, inf]; ...
                'w_T', ones(sizeT), [0, inf]; ...
                'w_swmax', ones(sizeS) * CV_pen, [0, inf]; ...
                'w_RV', RV_pen, [1e-3, inf]; ...
            };
            obj.weights.bound = cell2struct(pars(:, 3), pars(:, 1));
            obj.weights.value = cell2struct(pars(:, 2), pars(:, 1));

            % create also the symbolical weights struct
            obj.weights.sym.Q = struct;
            obj.weights.sym.V = struct;
            for name = pars(:, 1)'
                obj.weights.sym.Q.(name{1}) = obj.Q.pars.(name{1});
                obj.weights.sym.V.(name{1}) = obj.V.pars.(name{1});
            end
        end
    end
end
