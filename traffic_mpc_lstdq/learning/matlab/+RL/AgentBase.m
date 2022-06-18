classdef (Abstract) AgentBase < handle
    % AGENTBASE. An abstract base class for RL agents for traffic control 
    % with an MPC scheme as function approximator. 
    
    properties (GetAccess = public, SetAccess = protected)
        env (1, 1) % METANET.TrafficEnv 
        Q (1, 1) % MPC.NMPC
        V (1, 1) % MPC.NMPC
        %
        weights (1, 1) struct
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

            % compute symbolical derivatives
            % ...

            % TODO: add weight.sym as the vector of symbolical parameters
            % initialize learnable parameters/weights
            obj.init_pars(known_mdl_pars); 

            % prepare historical data containers
            % .... what to save in the base class ...?
        end
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
