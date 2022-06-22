classdef AgentMonitor < handle
    % AGENTMONITOR. OpenAI-like gym wrapper for MPC-based RL agents.
    
    properties
        agent (1, 1)                % RL.AgentBase
        %
        mpcruns (1, :) struct       % MPC history
        transitions (1, :) struct   % Transition history
        updates (1, :) struct       % Update history
        % 
        weights (1, :) struct       % history of agent's weight values
    end
    
    properties (Dependent)
        nb_updates (1, 1) double
    end



    methods (Access = public)
        function obj = AgentMonitor(agent)
            % AGENTMONITOR. Construct an empty instance of this class.
            arguments
                agent (1, 1) RL.AgentBase
            end
            obj.agent = agent;
            obj.clear_history();
        end
        
        function [r0_opt, sol, info] = solve_mpc(obj, name, varargin)
            [r0_opt, sol, info] = obj.agent.solve_mpc(name, varargin{:});
            
            % save: type, objval f, success
            s = struct('type', name, 'f', info.f, 'success', info.success);
            if isempty(obj.mpcruns)
                obj.mpcruns = s;
            else
                obj.mpcruns(end + 1) = s;
            end
        end

        function s = save_transition(obj, varargin)
            s = obj.agent.save_transition(varargin{:});

            % save: all the returned structure
            if isempty(obj.transitions)
                obj.transitions = s;
            else
                obj.transitions(end + 1) = s;
            end
        end

        function [n, deltas, s] = update(obj, varargin)
            % what to save: descent direction p, Hmod
            [n, deltas, s] = obj.agent.update(varargin{:});

            % save: all the returned structure
            if isempty(obj.updates)
                obj.updates = s;
            else
                obj.updates(end + 1) = s;
            end

            % save also the new weights
            obj.weights(end + 1) = obj.agent.weights.value;
        end

        function clear_history(obj)
            % CLEAR_HISTORY. Clears the agent history saved so far.

            % unlike the traffic monitor, we cannot easily preallocate 
            % nan arrays that will be then filled. In this case, when the
            % length of data to be saved is undefined, either use a cell 
            % array or append to an ever-growing array
            obj.mpcruns = struct.empty;
            obj.transitions = struct.empty;
            obj.updates = struct.empty;
            obj.weights = obj.agent.weights.value;
        end
    end

    % PROPERTY GETTERS
    methods 
        function nb = get.nb_updates(obj)
            nb = length(obj.updates);
        end
    end
end
