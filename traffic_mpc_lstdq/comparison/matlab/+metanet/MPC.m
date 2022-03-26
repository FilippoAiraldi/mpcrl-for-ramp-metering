classdef MPC < handle
    % MPC Wrapper around casadi.Opti to facilitate solving MPC problem for
    % the given 3-link metanet problem.
    

    %% PROPERTIES
    properties(GetAccess = public, SetAccess = private)
        Np
        Nc
        M
        a
        v_free
        rho_crit
        opti
        vars
        pars
    end


    %% PUBLIC METHODS
    methods (Access = public) 
        % constructor
        function obj = MPC(Np, Nc, M, Fdyn, a, v_free, rho_crit, T, L, lanes)
            % save variables
            obj.Np = Np;
            obj.Nc = Nc;
            obj.M = M;
            obj.a = a;
            obj.v_free = v_free;
            obj.rho_crit = rho_crit;

            % create opti
            obj = obj.create_opti(Fdyn, T, L, lanes);
        end

        function [w_opt, rho_opt, v_opt, r_opt, info] = solve(obj,...
                d, w0, rho0, v0, r0, w_last, rho_last, v_last, r_last)
            
            % set parameters
            obj.opti.set_value(obj.pars.d, d);
            obj.opti.set_value(obj.pars.w0, w0);
            obj.opti.set_value(obj.pars.rho0, rho0);
            obj.opti.set_value(obj.pars.v0, v0);
            obj.opti.set_value(obj.pars.r_last, r0);
        
            % warm start
            obj.opti.set_initial(obj.vars.w, w_last);
            obj.opti.set_initial(obj.vars.rho, rho_last);
            obj.opti.set_initial(obj.vars.v, v_last);
            obj.opti.set_initial(obj.vars.r, r_last);
            obj.opti.set_initial(obj.vars.slack, 0);

            % run solver
            try
                sol = obj.opti.solve();
                info = struct();
                get_value = @(o) sol.value(o);
            catch ME1
                try
                    stats = obj.opti.debug.stats();
                    info = struct('error', stats.return_status);
                    get_value = @(o) obj.opti.debug.value(o);
                catch ME2
                    rethrow(addCause(ME2, ME1))
                end   
            end

            % get outputs
            info.f = get_value(obj.opti.f);
            info.slack = get_value(obj.vars.slack);
            w_opt = get_value(obj.vars.w);
            rho_opt = get_value(obj.vars.rho);
            v_opt = get_value(obj.vars.v);
            r_opt = get_value(obj.vars.r);
        end
    end

    %% PRIVATE METHODS
    methods (Access = private)  
        function obj = create_opti(obj, F, T, L, lanes)
            % create opti stack
            obj.opti = casadi.Opti();

            % create vars and pars
            obj.vars = struct;
            obj.vars.w = obj.opti.variable(2, obj.M * obj.Np + 1);     
            obj.vars.rho = obj.opti.variable(3, obj.M * obj.Np + 1);   
            obj.vars.v = obj.opti.variable(3, obj.M * obj.Np + 1);      
            obj.vars.r = obj.opti.variable(1, obj.Nc);
            obj.vars.slack = obj.opti.variable(1, obj.M * obj.Np + 1);
            obj.pars = struct;
            obj.pars.d = obj.opti.parameter(3, obj.M * obj.Np);       
            obj.pars.w0 = obj.opti.parameter(2, 1);
            obj.pars.rho0 = obj.opti.parameter(3, 1);
            obj.pars.v0 = obj.opti.parameter(3, 1);
            obj.pars.r_last = obj.opti.parameter(1, 1);

            % cost to minimize
            cost = metanet.TTS(obj.vars.w, obj.vars.rho, T, L, lanes) + ...
                0.4 * metanet.input_variability_penalty(obj.pars.r_last, obj.vars.r) + ...
                1e1 * sum(obj.vars.slack, 2);
            obj.opti.minimize(cost);

            % constraints on domains
            obj.opti.subject_to(0.2 <= obj.vars.r <= 1) %#ok<CHAIN> 
            obj.opti.subject_to(obj.vars.w(:) >= 0)
            obj.opti.subject_to(obj.vars.rho(:) >= 0)
            obj.opti.subject_to(obj.vars.v(:) >= 0)
            
            % constraints on initial conditions
            obj.opti.subject_to(obj.vars.w(:, 1) == obj.pars.w0)
            obj.opti.subject_to(obj.vars.v(:, 1) == obj.pars.v0)
            obj.opti.subject_to(obj.vars.rho(:, 1) == obj.pars.rho0)

            % expand control sequence
            r_exp = [repelem(obj.vars.r, obj.M), ...
                repelem(obj.vars.r(end), obj.M * (obj.Np - obj.Nc))'];

            % constraints on state evolution
            for k = 1:obj.M * obj.Np
                [~, w_next, ~, rho_next, v_next] = F(obj.vars.w(:, k), ...
                    obj.vars.rho(:, k), obj.vars.v(:, k), r_exp(:, k), ...
                    obj.pars.d(:, k), obj.a, obj.v_free, obj.rho_crit);
                obj.opti.subject_to(obj.vars.w(:, k + 1) == w_next)
                obj.opti.subject_to(obj.vars.rho(:, k + 1) == rho_next)
                obj.opti.subject_to(obj.vars.v(:, k + 1) == v_next)
            end

            % custom constraints
            obj.opti.subject_to(obj.vars.slack(:) >= 0);
            obj.opti.subject_to(obj.vars.w(2, :) - obj.vars.slack <= 100);

            % set solver for opti
            plugin_opts = struct('expand', false, 'print_time', false);
            solver_opts = struct('print_level', 0, 'max_iter', 3e3);
            obj.opti.solver('ipopt', plugin_opts, solver_opts);
            % plugin_opts = struct('qpsol', 'osqp', 'expand', true, 'print_time', false);
            % obj.opti.solver('sqpmethod', plugin_opts);
        end
    end
end