function [init_cost, stage_cost, terminal_cost] = get_mpc_costs(model, ...
                    mpc, init_type, stage_type, terminal_type)
    % GET_MPC_COSTS. Returns the MPC's learnable initial cost, stage cost 
    % and terminal cost terms.
    arguments
        model (1, 1) struct 
        mpc (1, 1) struct 
        init_type, stage_type, terminal_type (1, :) char {mustBeTextScalar}
    end
    n_links = model.n_links;
    n_origins = model.n_origins;

    % create symbolic arguments
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    v = casadi.SX.sym('v', n_links, 1);
    rho_crit = casadi.SX.sym('rho_crit', 1, 1);
    v_free = casadi.SX.sym('v_free', 1, 1);
    w_norm = mpc.pars.normalization.w;
    rho_norm = mpc.pars.normalization.rho;
    v_norm = mpc.pars.normalization.v;



    %% initial cost
    switch init_type
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
        {'w', 'rho', 'v', 'weight'}, {init_type});
    assert(isequal(size(J0), [1, 1]))



    %% stage cost
    switch stage_type
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
        {stage_type});
    assert(isequal(size(Jk), [1, 1]))

    

    %% final/terminal cost
    switch terminal_type
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
        {stage_type});
    assert(isequal(size(JN), [1, 1]))
end
