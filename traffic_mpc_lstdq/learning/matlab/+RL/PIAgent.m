classdef PIAgent < RL.AgentBase
    % PIAGENT. Perfect-information, baseline agent.
    
    properties
        Property1
    end
    
    methods (Access = public)
        function obj = PIAgent(env, agentname)
            % PIAGENT. Constructs an instance of the MPC-based Q learning 
            % agent.
            arguments
                env (1, 1) METANET.TrafficEnv
                agentname (1, :) = char.empty
            end
            mdl_pars = struct('a', env.model.a, ... % pass the true values
                              'v_free', env.model.v_free, ...
                              'rho_crit', env.model.rho_crit);
            obj = obj@RL.AgentBase(env, mdl_pars, agentname);

            % set all other RL weights to zero
            for n = obj.weightnames
                if startsWith(n, 'w_')
                    obj.weights.value.(n) = ...
                                    zeros(size(obj.weights.value.(n)));
                end
            end

            % set constraint violation and rate variability cost to the 
            % same used in RL stage cost
            obj.weights.value.w_swmax = env.mpc.CV_penalty;
            obj.weights.value.w_RV = env.mpc.RV_penalty;
        end
    end

    methods (Hidden)
        function save_transition(~)
            error('Cannot save transitions for baseline agent.')
        end

        function update(~)
            error('Cannot perform updates for baseline agent.')
        end

        function set_weight_values(~)
            error('Cannot set weight values for baseline agent.')
        end
    end
end

