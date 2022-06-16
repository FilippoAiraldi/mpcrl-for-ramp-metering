function mpc = get_pars(model)
    % GET_PARS. Returns a structure containing the RL-MPC parameters.
    arguments
        model (1, 1) struct
    end
    K = model.K;

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
    mpc.update_freq = K / 2;        % when rl should update
    mpc.mem_cap = K / mpc.M * 10;   % RL experience replay capacity
    mpc.mem_sample = K / mpc.M * 5; % RL experience replay sampling size
    mpc.mem_last = K / mpc.M;       % percentage of last experiences to include in sample
    mpc.normalization.w = model.max_queue;
    mpc.normalization.rho = model.rho_max;
    mpc.normalization.v = model.v_free_wrong;
    mpc.normalization.r = model.C(2);

    % solver options
    mpc.multistart = 1; %4 * 4;             % multistarting NMPC solver
    mpc.opts_ipopt = struct('expand', 1, ...
                            'print_time', 0, ...
                            'ipopt', struct('print_level', 0, ...
                                            'max_iter', 3e3, ...
                                            'tol',1e-8, ...
                                            'barrier_tol_factor', 10));
end
