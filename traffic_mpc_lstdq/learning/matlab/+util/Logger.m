classdef Logger < handle
    % LOGGER. Class for logging purposes.
    
    properties (GetAccess = public, SetAccess = protected)
        env (1, 1) % METANET.TrafficMonitor
    end

    properties
       clock (1, 1) uint64 
    end

    methods (Access = public)
        function obj = Logger(env, runname, savetodiary)
            % LOGGER. Instantiates the logger class.
            arguments
                env (1, 1) METANET.TrafficMonitor
                runname (1, :) char {mustBeTextScalar} = char.empty
                savetodiary (1, 1) logical = true
            end
            obj.env = env;
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
        end

        function delete(~)
            % stop diary logging
            diary off;
        end
        
        function m = log(obj, msg)
            % LOG. Prints an information message to the command window.
            if isempty(msg)
                m = char.empty;
                return
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
            m = sprintf('[%s|%i|%i|%s] - [%s|%i|%.1f%%]', ...
                        time_tot, i, e, time_ep, ...
                        time_sim, k, k / K * 100);
        
            % add optional message
            if nargin > 1
                m = sprintf('%s - %s', m, msg);
            end
        
            % finally print
            fprintf('%s\n', m)
        end
    
        function m = log_mpc_status(obj, infoV, infoQ)
            % LOG_MPC_STATUS. Logs status related to solving V and Q mpcs.
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
    end

    methods (Static)
        function h = log_headers()
            h = '[Time_tot|Iter|Ep|Time_ep] - [Time_sim|k|Perc] - Message';
            fprintf(strcat(h, '\n'));
        end
    end
end
