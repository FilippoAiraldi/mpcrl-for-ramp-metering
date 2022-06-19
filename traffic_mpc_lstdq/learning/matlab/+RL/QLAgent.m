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

        function s = save_transition(obj, replaymem, L, infoQ, infoV)
            arguments
                obj (1, 1) RL.QLAgent
                replaymem (1, 1) RL.ReplayMem
                L (1, 1) double
                infoQ (1, 1) struct
                infoV (1, 1) struct
            end

            % if any of the two MPC failed, don't save to replaymem and 
            % return nans
            if ~infoQ.success || ~infoV.success
                % return nans
                td_err = nan;
                g = nan(size(obj.deriv.dQ));
            else
                % compute td error
                target = L + obj.env.mpc.discount * infoV.f;
                td_err = target - infoQ.f;
    
                % compute numerical gradients w.r.t. params
                dQ = infoQ.get_value(obj.deriv.dQ);
                d2Q = infoQ.get_value(obj.deriv.d2Q);
    
                % compute gradient and approximated hessian
                g = -td_err * dQ;
                H = dQ * dQ' - td_err * d2Q;
    
                % save to replay memory
                replaymem.add('g', g, 'H', H);
            end

            % create a struct for all transition quantities that need to 
            % be saved (avoid matrices, they are pretty expensive)
            s = struct('td_err', td_err, ...
                       'td_err_perc', td_err / infoQ.f, ...
                       'g', g); 
        end
    end
end

