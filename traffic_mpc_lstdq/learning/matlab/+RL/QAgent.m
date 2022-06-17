classdef QAgent < RL.MPCAgentBase
    % QAGENT. MPC-based RL agent for Q learning for traffic control.
    
    methods
        function obj = QAgent(env)
            % QAGENT. Constructs an instance of the MPC-based Q learning 
            % agent.
            arguments
                env (1, 1) METANET.TrafficEnv
            end
            
            % call superclass constructor
            obj = obj@RL.MPCAgentBase(env);
        end
    end
end

