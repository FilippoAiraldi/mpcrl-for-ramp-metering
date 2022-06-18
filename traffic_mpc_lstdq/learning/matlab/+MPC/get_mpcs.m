function [Q, V] = get_mpcs(sim, model, mpc, dynamics, TTS, RV)
    % GET_MPCS. Builds the NMPC problems Q and V. 
    arguments
        sim (1, 1) struct
        model (1, 1) struct
        mpc (1, 1) struct
        dynamics (1, 1) struct
        TTS (1, 1) casadi.Function
        RV (1, 1) casadi.Function
    end

    % get the objective function components
    [Vcost, Lcost, Tcost] = get_costs(model, mpc);
    
    % build mpc-based value function approximators
    for name = ["Q", "V"]
        % instantiate MPC controller
        C = MPC.NMPC(name, sim, model, mpc, dynamics);

        % create required parameters
        C.add_par('v_free_tracking', [1, 1]);
        C.add_par('weight_V', size(Vcost.mx_in( ...
                  find(startsWith(string(Vcost.name_in), 'weight')) - 1)));
        C.add_par('weight_L', size(Lcost.mx_in( ...
                  find(startsWith(string(Lcost.name_in), 'weight')) - 1)));
        C.add_par('weight_T', size(Tcost.mx_in( ...
                  find(startsWith(string(Tcost.name_in), 'weight')) - 1)));
        C.add_par('weight_rate_var', [1, 1]);
        C.add_par('weight_slack_w_max', [numel(C.vars.slack_w_max), 1]);
        C.add_par('r_last', [size(C.vars.r, 1), 1]);

        % initial, stage and terminal learnable costs
        J = Vcost(C.vars.w(:, 1), ...
                  C.vars.rho(:, 1), ...
                  C.vars.v(:, 1), ...
                  C.pars.weight_V);
        for k = 2:mpc.M * mpc.Np
            J = J + Lcost(C.vars.rho(:, k), ...
                          C.vars.v(:, k), ...
                          C.pars.rho_crit, ...
                          C.pars.v_free_tracking, ... % C.pars.v_free
                          C.pars.weight_L);
        end
        % terminal cost
        J = J + Tcost(C.vars.rho(:, end), ...
                      C.vars.v(:, end), ...
                      C.pars.rho_crit, ...
                      C.pars.v_free_tracking, ...
                      C.pars.weight_T);
        % max queue slack cost
        J = J + C.pars.weight_slack_w_max' * C.vars.slack_w_max(:);
        % traffic-related cost
        J = J ...
            + sum(TTS(C.vars.w, C.vars.rho)) ...                    % TTS
            + C.pars.weight_rate_var * RV(C.pars.r_last, C.vars.r); % terminal rate variability

        % assign cost to opti
        C.minimize(J);

        % case-specific modification
        if strcmp(name, "Q")
            % Q approximator has additional constraint on first action
            C.add_par('r0', [size(C.vars.r, 1), 1]);
            C.add_con('r0_blocked', C.vars.r(:, 1) - C.pars.r0, 0, 0);

            % assign to output
            Q = C;
        else
            % V approximator has perturbation to enhance exploration
            C.add_par('perturbation', size(C.vars.r(:, 1)));
            C.minimize(C.f + C.pars.perturbation' * ...
                                C.vars.r(:, 1) ./ mpc.normalization.r);

            % assign to output
            V = C;
        end
    end
end



%% local functions
function [init_cost, stage_cost, terminal_cost] = get_costs(model, mpc)
    % GET_MPC_COSTS. Returns the MPC's learnable initial cost, stage cost 
    % and terminal cost terms.
    arguments
        model (1, 1) struct 
        mpc (1, 1) struct 
    end
    n_links = model.n_links;
    n_origins = model.n_origins;

    % create symbolic arguments
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    v = casadi.SX.sym('v', n_links, 1);
    rho_crit = casadi.SX.sym('rho_crit', 1, 1);
    v_free = casadi.SX.sym('v_free', 1, 1);
    w_norm = mpc.normalization.w;
    rho_norm = mpc.normalization.rho;
    v_norm = mpc.normalization.v;



    %% initial cost
    switch mpc.cost_type.init
        case 'affine' 
            weight = casadi.SX.sym('weight', n_origins + 2 * n_links, 1);
            J0 = weight' * [w / w_norm; rho / rho_norm; v / v_norm];
        case 'constant'
            weight = casadi.SX.sym('weight', 1, 1);
            J0 = weight;
        otherwise
            error('Invalid init type (''affine'', ''constant'')')
    end
    init_cost = casadi.Function('init_cost', ...
        {w, rho, v, weight}, {J0}, ...
        {'w', 'rho', 'v', 'weight'}, {mpc.cost_type.init});
    assert(isequal(size(J0), [1, 1]))



    %% stage cost
    switch mpc.cost_type.stage
        case 'full'
            weight_rho = casadi.SX.sym('weight_rho', n_links, n_links);
            weight_v = casadi.SX.sym('weight_v', n_links, n_links);
            Q_rho = weight_rho;
            Q_v = weight_v;
        case 'diag'
            weight_rho = casadi.SX.sym('weight_rho', n_links, 1);
            weight_v = casadi.SX.sym('weight_v', n_links, 1);
            Q_rho = diag(weight_rho);
            Q_v = diag(weight_v);
        case 'posdef'
            weight_rho = casadi.SX.sym('weight_rho', n_links, 1);
            weight_v = casadi.SX.sym('weight_v', n_links, 1);
            Q_rho = diag(weight_rho.^2);
            Q_v = diag(weight_v.^2);
        otherwise
            error('Invalid stage type (''full'', ''diag'', ''posdef'')')
    end

    Jk = ((rho - rho_crit)' * Q_rho * (rho - rho_crit)) / rho_norm^2 ...
                    + ((v - v_free)' * Q_v * (v - v_free)) / v_norm^2;

    stage_cost = casadi.Function('stage_cost', ...
        {rho, v, rho_crit, v_free, [weight_rho; weight_v]}, ...
        {Jk}, {'rho', 'v', 'rho_crit', 'v_free', 'weight'}, ... 
        {mpc.cost_type.stage});
    assert(isequal(size(Jk), [1, 1]))

    

    %% final/terminal cost
    switch mpc.cost_type.terminal
        case 'full'
            weight_rho = casadi.SX.sym('weight_rho', n_links, n_links);
            weight_v = casadi.SX.sym('weight_v', n_links, n_links);
            Q_rho = weight_rho;
            Q_v = weight_v;
        case 'diag'
            weight_rho = casadi.SX.sym('weight_rho', n_links, 1);
            weight_v = casadi.SX.sym('weight_v', n_links, 1);
            Q_rho = diag(weight_rho);
            Q_v = diag(weight_v);
        case 'posdef'
            weight_rho = casadi.SX.sym('weight_rho', n_links, 1);
            weight_v = casadi.SX.sym('weight_v', n_links, 1);
            Q_rho = diag(weight_rho.^2);
            Q_v = diag(weight_v.^2);
        otherwise
            error('Invalid final type (''full'', ''diag'', ''posdef'')')
    end

    JN = ((rho - rho_crit)' * Q_rho * (rho - rho_crit)) / rho_norm^2 ...
                    + ((v - v_free)' * Q_v * (v - v_free)) / v_norm^2;

    terminal_cost = casadi.Function('terminal_cost', ...
        {rho, v, rho_crit, v_free, [weight_rho; weight_v]}, ...
        {JN}, {'rho', 'v', 'rho_crit', 'v_free', 'weight'}, ... 
        {mpc.cost_type.terminal});
    assert(isequal(size(JN), [1, 1]))
end
