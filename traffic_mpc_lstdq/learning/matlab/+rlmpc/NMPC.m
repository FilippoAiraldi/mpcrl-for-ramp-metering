classdef NMPC < handle
    % NMPC. Wrapper around casadi.Opti to facilitate solving MPC problem 
    % for the given 3-link metanet problem.
    

    properties (GetAccess = public, SetAccess = private)
        Np (1, 1) double 
        Nc (1, 1) double 
        M (1, 1) double
        opti (1, 1) % dont set class, otherwise an empty opti gets instantiated
        vars (1, 1) struct % contains opti variables
        pars (1, 1) struct % contains opti parameters
    end

    
    properties (GetAccess = private, SetAccess = private)
        eps (1, 1) double
    end


    methods (Access = public) 
        function obj = NMPC(Np, Nc, M, dynamics, eps)
            % NMPC. Builds an instance of an NMPC with the corresponding
            % horizons and dynamics.
            arguments
                Np, Nc, M (1, 1) double {mustBePositive,mustBeInteger}
                dynamics (1, 1) struct
                eps (1, 1) double {mustBeNonnegative} = 0
            end
            
            % create opti stack
            opti = casadi.Opti();

            % create necessary vars
            vars = struct;
            vars.w = opti.variable(dynamics.states.w.size(1), M * Np + 1);  % origin queue lengths  
            vars.rho = opti.variable( ...
                dynamics.states.rho.size(1), M * Np + 1);                   % link densities
            vars.v = opti.variable(dynamics.states.v.size(1), M * Np + 1);  % link speeds
            vars.r = opti.variable(dynamics.input.r.size(1), Nc);           % ramp metering rates

			% create necessary pars
            pars = struct;
            pars.d = opti.parameter(dynamics.dist.d.size(1), M * Np);       % origin demands
            pars.w0 = opti.parameter(dynamics.states.w.size(1), 1);         % initial values
            pars.rho0 = opti.parameter(dynamics.states.rho.size(1), 1);                   
            pars.v0 = opti.parameter(dynamics.states.v.size(1), 1);

            % create params for the system dynamics - might vary
            pars_dyn = {}; % will be helpful later to call dynamics.f
            for par = fieldnames(dynamics.pars)'
                sz = dynamics.pars.(par{1}).size;
                p = opti.parameter(sz(1), sz(2));
                pars.(par{1}) = p;
                pars_dyn{end + 1} = p; %#ok<AGROW> 
            end

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
                [~, w_next, ~, rho_next, v_next] = dynamics.f(...
                    vars.w(:, k), ...
                    vars.rho(:, k), ...
                    vars.v(:, k), ...
                    r_exp(:, k), ...
                    pars.d(:, k), ...
                    pars_dyn{:});
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
            obj.eps = eps;
        end

        function par = add_par(obj, name, dim1, dim2)
            % ADD_PAR. Adds a parameter to the Opti NMPC instance, with the
            % given name and size.
            arguments
                obj (1, 1) rlmpc.NMPC
                name (1, :) char {mustBeTextScalar}
                dim1 (1, :) double {mustBePositive,mustBeInteger}
                dim2 (1, 1) double {mustBePositive,mustBeInteger} = 1
            end
            assert(all(~strcmp(fieldnames(obj.pars), name), 'all'), ...
                'parameter name already in use')
            assert(nargin == 4 || ~isscalar(dim1), ...
                'specify dimensions in two scalars or a single matrix')
            if nargin < 4
                dim2 = dim1(2);
                dim1 = dim1(1);
            end
            par = obj.opti.parameter(dim1, dim2);
            obj.pars.(name) = par;
        end

        function var = add_var(obj, name, dim1, dim2)
            % ADD_VAR. Adds a variable to the Opti NMPC instance, with the
            % given name and size.
            arguments
                obj (1, 1) rlmpc.NMPC
                name (1, :) {mustBeTextScalar}
                dim1 (1, :) {mustBePositive,mustBeInteger}
                dim2 (1, 1) {mustBePositive,mustBeInteger} = 1
            end
            assert(all(~strcmp(fieldnames(obj.vars), name), 'all'), ...
                'variable name already in use')
            assert(nargin == 4 || ~isscalar(dim1), ...
                'specify dimensions in two scalars or a single matrix')
            if nargin < 4
                dim2 = dim1(2);
                dim1 = dim1(1);
            end
            var = obj.opti.variable(dim1, dim2);
            obj.vars.(name) = var;
        end

        function [sol, info] = solve(obj, pars, vals)
            % SOLVE. Solve the NMPC problem with the given parameter values
            % and initial conditions for the variables.
            arguments
                obj (1, 1) rlmpc.NMPC
                pars (1, :) struct
                vals (1, :) struct
            end
            assert(all(pars.w0 >= obj.eps) && all(pars.rho0 >= obj.eps) ...
                && all(pars.v0 >= obj.eps), 'infeasible init. conditions')

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
                info.sol = s;
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
            % info.g = get_value(obj.opti.g);
            % info.lam_g = get_value(obj.opti.lam_g);
            sol = struct;
            for name = fieldnames(obj.vars)'
                sol.(name{1}) = full(get_value(obj.vars.(name{1})));
            end
        end

        function v = concat_pars(obj, names)
            % GROUP_PARS. Concatenates in a single vertical array the NMPC 
            % parameters whose name appears in the list.
            arguments
                obj (1, 1) rlmpc.NMPC
                names (:, 1) cell
            end
            v = cellfun(@(n) obj.pars.(n), names, 'UniformOutput', false);
            v = vertcat(v{:});
        end
    end
end
