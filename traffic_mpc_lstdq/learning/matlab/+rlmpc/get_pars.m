function mpc = get_pars(ep_length)
    % GET_PARS. Returns a structure containing the RL-MPC parameters.
    arguments
        ep_length (1, 1) double {mustBePositive} % length of a single episode
    end

    % horizons and horizon multipliers
    mpc.Np = 4;                             % prediction horizon - \approx 3*L/(M*T*v_avg)
    mpc.Nc = 3;                             % control horizon
    mpc.M  = 6;                             % horizon spacing factor

    % objective weights
    mpc.perturb_mag = 100;                  % magnitude of exploratory perturbation
    mpc.rate_var_penalty = 4e-2;            % penalty weight for rate variability
    mpc.con_violation_penalty = 10;         % penalty for constraint violations

    % RL parameters and update rules
    mpc.discount = 0.99;                    % rl discount factor
    mpc.lr0 = 1e-3;                         % fixed rl learning rate
    mpc.max_delta = 1 / 5;                  % percentage of maximum parameter change in a single update
    mpc.update_freq = ep_length / 2;        % when rl should update
    mpc.mem_cap = ep_length / mpc.M * 10;   % RL experience replay capacity
    mpc.mem_sample = ep_length / mpc.M * 5; % RL experience replay sampling size
    mpc.mem_last = ep_length / mpc.M;       % percentage of last experiences to include in sample

    % solver options
    mpc.multistart = 1; %4 * 4;             % multistarting NMPC solver
    mpc.opts_ipopt = struct('expand', 1, ...
                            'print_time', 0, ...
                            'ipopt', struct('print_level', 0, ...
                                            'max_iter', 3e3, ...
                                            'tol',1e-8, ...
                                            'barrier_tol_factor', 10));
end
