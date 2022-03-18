function log = logging(...
    realtime_tot, ep_n, realtime_ep, sim_time, sim_iter, sim_length, msg)

    % convert floats to durations
    realtime_tot = duration(0, 0, realtime_tot, 'Format', 'hh:mm:ss');
    realtime_ep = duration(0, 0, realtime_ep, 'Format', 'hh:mm:ss');
    sim_time = duration(sim_time, 0, 0, 'Format', 'hh:mm:ss');

    % assemble log string
    log = sprintf('[%s|%i|%s] - [%s|%i|%.1f%%]', ...
        realtime_tot, ep_n, realtime_ep, ...
        sim_time, sim_iter, sim_iter / sim_length * 100);

    % add optional message
    if nargin > 6
        log = sprintf('%s - %s', log, msg);
    end

    % finally print
    fprintf('%s\n', log)
end