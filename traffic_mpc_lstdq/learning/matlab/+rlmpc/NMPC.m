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
        function obj = NMPC(Np, Nc, M, Fdyn, eps)
            if nargin < 5
                eps = 0; % nonnegative constraint precision
            end

            % get some dimensions
            % F:(w[2],rho[3],v[3],r,d[2],a,v_free,rho_crit)->(q_o[2],w_o_next[2],q[3],rho_next[3],v_next[3])
            n_links = size(Fdyn.mx_in(1), 1);   % number of links
            n_orig = size(Fdyn.mx_in(0), 1);    % number of origins
            n_ramps = size(Fdyn.mx_in(3), 1);   % number of onramps
            n_d = size(Fdyn.mx_in(4), 1);       % number of disturbances
            
            % create opti stack
            opti = casadi.Opti();

            % create vars
            vars = struct;
            vars.w = opti.variable(n_orig, M * Np + 1);     % origin queue lengths  
            vars.rho = opti.variable(n_links, M * Np + 1);  % link densities
            vars.v = opti.variable(n_links, M * Np + 1);    % link speeds
            vars.r = opti.variable(n_ramps, Nc);            % ramp metering rates

			% create pars
            pars = struct;
            pars.d = opti.parameter(n_d, M * Np);           % origin demands    
            pars.w0 = opti.parameter(n_orig, 1);            % initial values
            pars.rho0 = opti.parameter(n_links, 1);                   
            pars.v0 = opti.parameter(n_links, 1);

            % params for the system dynamics
            pars.a = opti.parameter(1, 1);
            pars.v_free = opti.parameter(1, 1);
            pars.rho_crit = opti.parameter(1, 1);

            % constraints on domains
            opti.subject_to(-vars.r + 0.2 <= 0)
            opti.subject_to(vars.r - 1 <= 0)
            opti.subject_to(-vars.w(:) + eps <= 0)
            opti.subject_to(-vars.rho(:) + eps <= 0)
            opti.subject_to(-vars.v(:) + eps <= 0)
            
            % constraints on initial conditions
            opti.subject_to(vars.w(:, 1) - pars.w0 == 0)
            opti.subject_to(vars.v(:, 1) - pars.v0 == 0)
            opti.subject_to(vars.rho(:, 1) - pars.rho0 == 0)

            % expand control sequence
            r_exp = [repelem(vars.r, 1, M), ...
                repelem(vars.r(:, end), 1, M * (Np - Nc))];

            % constraints on state evolution
            for k = 1:M * Np
                [~, w_next, ~, rho_next, v_next] = Fdyn(...
                    vars.w(:, k), ...
                    vars.rho(:, k), ...
                    vars.v(:, k), ...
                    r_exp(:, k), ...
                    pars.d(:, k), ...
                    pars.a, ...
                    pars.v_free, ...
                    pars.rho_crit);
                opti.subject_to(vars.w(:, k + 1) - w_next == 0)
                opti.subject_to(vars.rho(:, k + 1) - rho_next == 0)
                opti.subject_to(vars.v(:, k + 1) - v_next == 0)
            end

            % save to instance
            obj.Np = Np;
            obj.Nc = Nc;
            obj.M = M;
            obj.opti = opti;
            obj.vars = vars;
            obj.pars = pars;
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