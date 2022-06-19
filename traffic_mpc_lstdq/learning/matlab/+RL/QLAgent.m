classdef QLAgent < RL.AgentBase
    % QLAGENT. MPC-based RL agent for Q learning for traffic control.
    
    properties (GetAccess = public, SetAccess = protected)
        deriv (1, 1) struct
    end

    methods (Access = public)
        function obj = QLAgent(env, known_mdl_pars)
            % QLAGENT. Constructs an instance of the MPC-based Q learning 
            % agent.
            arguments
                env (1, 1) METANET.TrafficEnv
                known_mdl_pars (1, 1) struct
            end
            
            % call superclass constructor
            obj = obj@RL.AgentBase(env, known_mdl_pars);

            % compute derivative and approx. hessian of Q
            w = obj.w_sym('Q');
            Lagr = obj.Q.lagrangian;
            obj.deriv = struct('dQ', simplify(jacobian(Lagr, w)'), ...
                               'd2Q', simplify(hessian(Lagr, w))); % approx
        end

        function GET_V(obj, pars, sol0)
            % GET_V. Computes the value function V(s) at the given state 
            % (included in pars) by running the underlying MPC scheme. 
            arguments
                obj (1, 1) RL.QAgent
                pars (1, 1) struct
                sol0 (1, 1) = [] % struct
            end

            % if provided, use sol0 to warmstart the MPC. If not provided,
            % build it from the given state.

            % set the MPC parameters

            % call the MPC 

            % return

        end

        function run_Q(obj, state, r, sol0)
            % GET_Q. Computes the action function Q(s, a) at the given 
            % state by running the underlying MPC scheme.             
        end
    end
end

