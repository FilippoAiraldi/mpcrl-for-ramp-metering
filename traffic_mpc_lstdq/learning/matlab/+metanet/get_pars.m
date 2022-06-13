function mdl = get_pars()
    % GET_PARS. Returns a structure containing the METANET model 
    % parameters.
    
    % simulation paramaters
    mdl.Tfin = 2;                       % simulation time per episode (h)
    mdl.T = 10 / 3600;                  % simulation step size (h)
    mdl.K = mdl.Tfin / mdl.T;           % simulation steps per episode
    mdl.t = (0:(mdl.K - 1)) * mdl.T;    % time vector (h)

    % network size
    mdl.n_origins = 2;                  % number of origins
    mdl.n_links = 3;                    % number of links
    mdl.n_ramps = 1;                    % number of controlled on-ramps
    mdl.n_dist = 3;                     % number of disturbances/demands

    % segments
    mdl.L = 1;                          % length of links (km)
    mdl.lanes = 2;                      % lanes per link (adim)
    
    % origins O1 and O2
    mdl.C = [3500, 2000];               % on-ramp capacity (veh/h/lane)
    mdl.max_queue = 50;                 % maximum queue (veh) constraint
    
    % model parameters
    mdl.tau = 18 / 3600;                % model parameter (s)
    mdl.kappa = 40;                     % model parameter (veh/km/lane)
    mdl.eta = 60;                       % model parameter (km^2/lane)
    mdl.rho_max = 180;                  % maximum capacity (veh/km/lane)
    mdl.delta = 0.0122;                 % merging phenomenum parameter
    
    % true (unknown) model parameters
    mdl.a = 1.867;                      % model parameter (adim)
    mdl.v_free = 102;                   % free flow speed (km/h)
    mdl.rho_crit = 33.5;                % critical capacity (veh/km/lane)

    % known (wrong) model parameters
    mdl.a_wrong = mdl.a * 1.3;
    mdl.v_free_wrong = mdl.v_free * 1.3;
    mdl.rho_crit_wrong = mdl.rho_crit * 0.7;
end
