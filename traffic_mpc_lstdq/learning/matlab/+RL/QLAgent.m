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
    end
end

