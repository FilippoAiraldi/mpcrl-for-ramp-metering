classdef QAgent < RL.AgentBase
    % QAGENT. MPC-based RL agent for Q learning for traffic control.
    
    methods
        function obj = QAgent(env, known_mdl_pars)
            % QAGENT. Constructs an instance of the MPC-based Q learning 
            % agent.
            arguments
                env (1, 1) METANET.TrafficEnv
                known_mdl_pars (1, 1) struct
            end
            
            % call superclass constructor
            obj = obj@RL.AgentBase(env, known_mdl_pars);
        end
    end
end

