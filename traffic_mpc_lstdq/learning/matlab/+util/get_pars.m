function [sim, mdl, mpc] = get_pars()
    % GET_PARS. Returns 3 structures containing 
    %   1. the simulation parameters,
    %   2. the METANET model parameters,
    %   3. the MPC design parameters.
    
    %% SIMULATION
    sim = struct;

    % simulation paramaters
    sim.Tfin = 2;                       % simulation time per episode (h)
    sim.T = 10 / 3600;                  % simulation step size (h)
    sim.K = sim.Tfin / sim.T;           % simulation steps per episode
    sim.t = (0:(sim.K - 1)) * sim.T;    % time vector (h)

    

    %% MODEL
    mdl = struct; 

    % network size
    mdl.n_links = 3;                    % number of links
    mdl.n_origins = 2;                  % number of origins
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



    %% MPC
    % horizons and horizon multipliers
    mpc.Np = 4;                             % prediction horizon - \approx 3*L/(M*T*v_avg)
    mpc.Nc = 3;                             % control horizon
    mpc.M  = 6;                             % horizon spacing factor

    % objective weights
    mpc.perturb_mag = 100;                  % magnitude of exploratory perturbation
    mpc.RV_penalty = 4e-2;                  % penalty weight for rate variability
    mpc.CV_penalty = 10;                    % penalty for constraint violations

    % types of cost terms
    mpc.cost_type.init = 'affine';
    mpc.cost_type.stage = 'diag';
    mpc.cost_type.terminal = 'diag';

    % RL parameters and update rules
    mpc.discount = 0.99;                    % rl discount factor
    mpc.lr0 = 1e-3;                         % fixed rl learning rate
    mpc.max_delta = 1 / 5;                  % percentage of maximum parameter change in a single update
    mpc.update_freq = sim.K / 2;            % when rl should update
    mpc.mem_cap = sim.K / mpc.M * 10;       % RL experience replay capacity
    mpc.mem_sample = sim.K / mpc.M * 5;     % RL experience replay sampling size
    mpc.mem_last = sim.K / mpc.M;           % percentage of last experiences to include in sample
    mpc.norm.w = mdl.max_queue;             % values to normalize each quantity
    mpc.norm.rho = mdl.rho_max;
    mpc.norm.v = mdl.v_free * 1.3;
    mpc.norm.r = mdl.C(2);

    % solver options
    mpc.multistart = 1; %4 * 4;             % multistarting NMPC solver
    mpc.opts_ipopt = struct('expand', 1, ...
                            'print_time', 0, ...
                            'ipopt', struct('print_level', 0, ...
                                            'max_iter', 3e3, ...
                                            'tol',1e-8, ...
                                            'barrier_tol_factor', 10));



    %% some checks
    assert(mod(sim.K, mpc.M) == 0)
end
