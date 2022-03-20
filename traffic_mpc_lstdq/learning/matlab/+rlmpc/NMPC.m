classdef NMPC < handle
    % MPC Wrapper around casadi.Opti to facilitate solving MPC problem for
    % the given 3-link metanet problem.
    

    properties (GetAccess = public, SetAccess = public)
        Np
        Nc
        M
        opti
        vars % contains opti variables
        pars % contains opti parameters
    end


    methods (Access = public) 
        function obj = NMPC(Np, Nc, M)
            obj.Np = Np;
            obj.Nc = Nc;
            obj.M = M;
            obj.opti = casadi.Opti();
        end

        function init_opti(obj, Fdynamics)
            % create vars
            obj.vars = struct;
            obj.vars.w = obj.opti.variable(2, obj.M * obj.Np + 1);      % origin queue lengths  
            obj.vars.r = obj.opti.variable(1, obj.Nc);                  % ramp metering rates
            obj.vars.rho = obj.opti.variable(3, obj.M * obj.Np + 1);    % link densities
            obj.vars.v = obj.opti.variable(3, obj.M * obj.Np + 1);      % link speeds

            % create pars
            obj.pars = struct;
            obj.pars.d = obj.opti.parameter(2, obj.M * obj.Np);         % origin demands    
            obj.pars.w0 = obj.opti.parameter(2, 1);                     % initial values
            obj.pars.rho0 = obj.opti.parameter(3, 1);                   
            obj.pars.v0 = obj.opti.parameter(3, 1);

            % params for the system dynamics
            obj.pars.a = obj.opti.parameter(1, 1);
            obj.pars.v_free = obj.opti.parameter(1, 1);
            obj.pars.rho_crit = obj.opti.parameter(1, 1);

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
                [~, w_next, ~, rho_next, v_next] = Fdynamics(...
                    obj.vars.w(:, k), ...
                    obj.vars.rho(:, k), ...
                    obj.vars.v(:, k), ...
                    r_exp(:, k), ...
                    obj.pars.d(:, k), ...
                    obj.pars.a, ...
                    obj.pars.v_free, ...
                    obj.pars.rho_crit);
                obj.opti.subject_to(obj.vars.w(:, k + 1) == w_next)
                obj.opti.subject_to(obj.vars.rho(:, k + 1) == rho_next)
                obj.opti.subject_to(obj.vars.v(:, k + 1) == v_next)
            end
        end

        function set_ipopt_opts(obj, plugin_opts, solver_opts)
            obj.opti.solver('ipopt', plugin_opts, solver_opts);
        end

        function set_cost(obj, cost)
            obj.opti.minimize(cost) 
        end

        function par = add_par(obj, name, nrows, ncols)
            assert(all(~strcmp(fieldnames(obj.pars), name), 'all'), ...
                'parameter name already in use')
            if nargin < 4
                ncols = nrows(2);
                nrows = nrows(1);
            end
            par = obj.opti.parameter(nrows, ncols);
            obj.pars.(name) = par;
        end

        function var = add_var(obj, name, nrows, ncols)
            assert(all(~strcmp(fieldnames(obj.vars), name), 'all'), ...
                'variable name already in use')
            if nargin < 4
                ncols = nrows(2);
                nrows = nrows(1);
            end
            var = obj.opti.variable(nrows, ncols);
            obj.vars.(name) = var;
        end

        function [sol, info] = solve(obj, pars, vals)
            % set parameter values
            for name = fieldnames(obj.pars)'
                obj.opti.set_value(obj.pars.(name{1}), pars.(name{1}));
            end

            % set initial conditions for the solver
            for name = fieldnames(obj.vars)'
                obj.opti.set_initial(obj.vars.(name{1}), vals.(name{1}));
            end

            % run solver
            info = struct;
            try
                s = obj.opti.solve();
                info.success = true;
                info.sol_obj = s;
                get_value = @(o) s.value(o);
            catch ME1
                try
                    stats = obj.opti.debug.stats();
                    info.success = false;
                    info.error = stats.return_status;
                    get_value = @(o) obj.opti.debug.value(o);
                catch ME2
                    rethrow(addCause(ME2, ME1))
                end   
            end

            % get outputs
            info.f = get_value(obj.opti.f);
%             info.g = get_value(obj.opti.g);
%             info.lam_g = get_value(obj.opti.lam_g);
            sol = struct;
            for name = fieldnames(obj.vars)'
                sol.(name{1}) = full(get_value(obj.vars.(name{1})));
            end
        end
    end
end