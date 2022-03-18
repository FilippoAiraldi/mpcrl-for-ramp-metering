function [init_cost, stage_cost, terminal_cost] = get_learnable_costs(...
    n_origins, n_links, ...
    w_max, rho_max, v_max, ...
    T, L, lanes, ...
    init_type, stage_type, final_type)


    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    v = casadi.SX.sym('v', n_links, 1);
    rho_crit = casadi.SX.sym('rho_crit', 1, 1);
    v_free = casadi.SX.sym('v_free', 1, 1);


    %% initial cost
    switch init_type
        case 'affine' 
            weight = casadi.SX.sym('weight', n_origins + 2 * n_links, 1);
            J0 = weight' * [w / w_max; rho / rho_max; v / v_max];
        case 'constant'
            weight = casadi.SX.sym('weight', 1, 1);
            J0 = weight;
        otherwise
            error('Invalid init type (''affine'', ''constant'')')
    end

    init_cost = casadi.Function('init_cost', ...
        {w, rho, v, weight}, {J0}, ...
        {'w', 'rho', 'v', 'weight'}, {'J_affine'});
    assert(isequal(size(J0), [1, 1]))


    %% stage cost
    % total time spent in traffic cost (not normalized!!)
    TTS = metanet.TTS(w, rho, T, L, lanes);

    % learnable cost term
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
    yr = (rho - rho_crit) / rho_max;
    yv = (v - v_free) / v_max;
    Jk = yr' * Q_rho * yr + yv' * Q_v * yv;

    % sum the components together
    stage_cost = casadi.Function('stage_cost', ...
        {w, rho, v, rho_crit, v_free, weight_rho, weight_v}, {Jk + TTS}, ...
        {'w', 'rho', 'v', 'rho_crit', 'v_free', 'weight_rho', 'weight_v'}, ... 
        {sprintf('J_%s', final_type)});
    assert(isequal(size(Jk), [1, 1]))


    %% final/terminal cost
    % create weights
    switch final_type
        case 'full'
            weight = casadi.SX.sym('weight', n_links, n_links);
            Q = weight;
        case 'diag'
            weight = casadi.SX.sym('weight', n_links, 1);
            Q = diag(weight);
        case 'posdef'
            weight = casadi.SX.sym('weight', n_links, 1);
            Q = diag(weight.^2);
        otherwise
            error('Invalid final type (''full'', ''diag'', ''posdef'')')
    end

    % compute cost and create function
    yr = (rho - rho_crit) / rho_max;
    JN = yr' * Q * yr;

    terminal_cost = casadi.Function('terminal_cost', ...
        {rho, rho_crit, weight}, {JN}, ...
        {'rho', 'rho_crit', 'weight'}, {sprintf('J_%s', final_type)});
    assert(isequal(size(JN), [1, 1]))
end