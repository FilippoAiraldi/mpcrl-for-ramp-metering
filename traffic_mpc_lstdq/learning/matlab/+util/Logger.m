classdef Logger < handle
    % LOGGER. Class for logging purposes.
    
    properties (GetAccess = public, SetAccess = protected)
        env (1, 1)      % METANET.TrafficMonitor
        agent (1, 1)    % RL.AgentMonitor
    end

    properties
       clock (1, 1) uint64 
    end

    properties (Access = private)
        logname (1, :) char = char.empty;
        max_headers_len (1, 1) double 
    end



    methods (Access = public)
        function obj = Logger(env, agent, runname, savetodiary)
            % LOGGER. Instantiates the logger class.
            arguments
                env (1, 1) METANET.TrafficMonitor
                agent (1, 1) RL.AgentMonitor
                runname (1, :) char {mustBeTextScalar} = char.empty
                savetodiary (1, 1) logical = true
            end
            obj.env = env;
            obj.agent = agent;
            obj.clock = tic;

            % start diary logging
            if savetodiary
                if ~isempty(runname)
                    diary(strcat(runname, '_log.txt'))
                else
                    diary on
                end
            end
            obj.log_headers();

            % compute max length of the log headers
            obj.max_headers_len = ...
                length('hh:mm:ss') * 3 + ...         % all time formats
                ceil(log10(env.iterations)) + ...    % integers
                ceil(log10(env.env.episodes)) + ...
                ceil(log10(env.env.sim.K)) + ...
                4 + ...                              % percentage 100%
                length('[|||] - [||]');              % separators
        end

        function delete(~)
            % stop diary logging
            diary off;
        end
        
        function m = log(obj, msg)
            % LOG. Prints an information message to the command window.
            arguments
                obj (1, 1) util.Logger
                msg (1, :) char
            end

            time_tot = toc(obj.clock);
            i = obj.env.iter;
            e = obj.env.env.ep;
            time_ep = obj.env.current_ep_time;
            k = obj.env.env.k;
            time_sim = obj.env.env.sim.t(k);
            K = obj.env.env.sim.K;

            % convert floats to durations
            time_tot = duration(0, 0, time_tot, 'Format', 'hh:mm:ss');
            time_ep = duration(0, 0, time_ep, 'Format', 'hh:mm:ss');
            time_sim = duration(time_sim, 0, 0, 'Format', 'hh:mm:ss');
        
            % assemble log string
            m = sprintf('[%s|%i|%i|%s] - [%s|%i|%.0f%%]', ...
                        time_tot, i, e, time_ep, ...
                        time_sim, k, k / K * 100);
       
            % add agent's name (optional) and the message
            filler = repmat('-', 1, obj.max_headers_len - length(m));
            n = obj.agent.agent.name;
            if isempty(n)
                m = sprintf('%s -%s %s', m, filler, msg);
            else
                m = sprintf('%s -%s %s - %s', m, filler, n, msg);
            end
                    
            % finally print
            fprintf('%s\n', m)
        end
    
        function m = log_mpc_status(obj, infoQ, infoV)
            % LOG_MPC_STATUS. Logs status related to solving V and Q mpcs.
            arguments
                obj (1, 1) util.Logger
                infoQ (1, 1) struct
                infoV (1, 1) struct
            end
            if infoV.success
                msg = '';
            else
                msg = sprintf('V: %s. ', infoV.msg);
            end
            if ~infoQ.success
                msg = sprintf('%sQ: %s.', msg, infoQ.msg);
            end
            m = obj.log(msg);
        end

        function m = log_ep_summary(obj, ep)
            % LOG_EP_SUMMARY. Logs a recap of the episode just finished 
            % (assumes the agent has not been yet reset).
            arguments
                obj (1, 1) util.Logger
                ep (1, 1) double {mustBeInteger, mustBePositive}
            end
            a = obj.agent.agent;

            fails = a.Q.failures + a.V.failures;
            m = obj.log(sprintf( ...
                'episode %i done: J=%.3f, TTS=%.3f, failures=%i', ...
                ep, obj.env.env.cumcost.J,obj.env.env.cumcost.TTS, fails));
        end

        function m = log_agent_update(obj, Nsamples)
            % LOG_AGENT_UPDATE. Logs the agent's weights after a new 
            % update.
            arguments
                obj (1, 1) util.Logger
                Nsamples (1, 1) double
            end
            a = obj.agent.agent;

            nb_update = obj.agent.nb_updates + 1;
            names = fieldnames(a.weights.value);
            values = cellfun(@(v) mat2str(v(:)', 6), ...
                             struct2cell(a.weights.value), ...
                             'UniformOutput', false);
            fmt = ['update %i (N=%i):', ...
                   repmat(' %s=%s;', 1, length(names))];
            rlargs = reshape([names, values]', [], 1);
            msg = sprintf(fmt, nb_update, Nsamples, rlargs{:});

            m = obj.log(msg);
        end
    end

    methods (Static)
        function h = log_headers()
            % LOG_HEADERS. Logs the headers of each column.
            h = '[Time_tot|Iter|Ep|Time_ep] - [Time_sim|k|Perc] - Message';
            fprintf(strcat(h, '\n'));
        end
    end
end
