classdef (Abstract) MPCAgentBase < handle
    % MPCAGENTBASE. An abstract base class for RL agents for traffic 
    % control with an MPC scheme as function approximator. 
    
    properties (GetAccess = public, SetAccess = protected)
        env (1, 1) % METANET.TrafficEnv 
        Q (1, 1) % MPC.NMPC
        V (1, 1) % MPC.NMPC
    end



    methods
        function obj = MPCAgentBase(env)
            arguments
                env (1, 1) METANET.TrafficEnv
            end
            
            % create the NMPC instances for Q(s,a) and V(s)
            obj.env = env;
            [obj.Q, obj.V] = MPC.get_mpcs(env.sim, env.model, env.mpc, ...
                                    env.dynamics, env.TTS, env.Rate_Var);

            % compute symbolical derivatives
            % ...

            % prepare historical data containers
            % .... what to save in the base class ...?
        end
    end
end
