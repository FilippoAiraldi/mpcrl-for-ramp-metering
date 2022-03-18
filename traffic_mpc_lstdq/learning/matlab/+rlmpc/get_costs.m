function costs = get_costs()
    % costs = GET_COSTS() 
    %   Returns the various terms composing the mpc and rl costs as casadi
    %   Functions.

    % mpc init cost
    mpc_init = 0;

    % mpc stage cost
    mpc_stage = 0;

    % mpc final cost
    mpc_final = 0;

    % rl stage cost
    rl_stage = 0;

    % return
    costs.mpc.initial = mpc_init;
    costs.mpc.stage = mpc_stage;
    costs.mpc.final = mpc_final;
    costs.rl.stage = rl_stage;
end

