function [origins, links, slack, objective, last_sol] = run_one_episode(K, mpc, init_conds, last_sol, replaymem)
    % RUN_ONE_EPISODE Runs one episode of MPC-controlled traffic network
    %   simulation.
    
    M = mpc.M;
    Np = mpc.Np;
    Nc = mpc.Nc;

    % initialize containers for results
    origins.queue = nan(size(mpc.vars.w, 1), K + 1);        
    origins.flow = nan(size(mpc.vars.w, 1), K);          
    origins.rate = nan(size(mpc.vars.r, 1), K);
    links.flow = nan(size(mpc.vars.v, 1), K);
    links.density = nan(size(mpc.vars.rho, 1), K + 1);
    links.speed = nan(size(mpc.vars.v, 1), K + 1);
    slack = nan(size(mpc.vars.slack, 2), K);
    objective = nan(1, K); 

    % set initial conditions
    origins.queue(:, 1) = init_conds.w0;
    links.density(:, 1) = init_conds.rho0;
    links.speed(:, 1) = init_conds.v0;

    % create last mpc solution
    if isempty(last_sol)
        w_last = repmat(init_conds.w0, 1, M * Np + 1);
        rho_last = repmat(init_conds.rho0, 1, M * Np + 1);
        v_last = repmat(init_conds.v0, 1, M * Np + 1);
        r_last = repmat(init_conds.r, 1, Nc);
    else
        w_last = {repmat(w{1}(:, 1), 1, M * Np + 1),     repmat(w{2}(:, 1), 1, M * Np + 1)};
        rho_last = {repmat(rho{1}(:, 1), 1, M * Np + 1), repmat(rho{2}(:, 1), 1, M * Np + 1)};
        v_last = {repmat(v{1}(:, 1), 1, M * Np + 1),     repmat(v{2}(:, 1), 1, M * Np + 1)};
        r_last = {repmat(r{1}(:, 1), 1, Nc),             repmat(r{2}(:, 1), 1, Nc)};
    end

end

