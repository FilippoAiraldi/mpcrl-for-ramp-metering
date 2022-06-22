classdef QLAgent < RL.AgentBase
    % QLAGENT. OpenAI-like gym MPC-based RL agent for Q learning for 
    % traffic control.
    
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

        function [n, deltas, s] = update(obj, replaymem)
            arguments   
                obj (1, 1) RL.QLAgent
                replaymem (1, 1) RL.ReplayMem
            end
            mem_sample = obj.env.mpc.mem_sample;
            mem_last = obj.env.mpc.mem_last;
            lr = obj.env.mpc.lr0;
            max_delta = obj.env.mpc.max_delta;

            % draw a sample batch of transitions
            sample = replaymem.sample(mem_sample, mem_last);
            n = sample.n;

            % compute descend direction
            [p, Hmod] = RL.descent_direction(sample.g, sample.H, 1);

            % perform constrained update and save its maximum multiplier
            [obj.weights.value, deltas] = RL.constr_update( ...
                            obj.weights.value, obj.weights.bound, ...
                            p, lr, max_delta);
            
            % build a struct with some information on the update. These 
            % will most likely be saved to a log history
            s = struct('n', n, 'p', p, 'Hmod', Hmod);
        end
    end
end

