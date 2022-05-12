classdef NMPC < handle
    % NMPC. Wrapper around casadi.Opti to facilitate solving MPC problem 
    % for the given 3-link metanet problem.
    


    properties (GetAccess = public, SetAccess = private)
        Np (1, 1) double 
        Nc (1, 1) double 
        M (1, 1) double
        dynamics (1, 1) struct
        max_queue (:, 1) double
        opti (1, 1) % dont set class, otherwise an empty opti gets instantiated
        vars (1, 1) struct % contains opti variables
        pars (1, 1) struct % contains opti parameters
        eps (1, 1) double
        % 
        con (1, 1) struct
    end



    methods (Access = public) 
        function obj = NMPC(Np, Nc, M, dynamics, max_queue, eps)
            % NMPC. Builds an instance of an NMPC with the corresponding
            % horizons and dynamics.
            arguments
                Np, Nc, M (1, 1) double {mustBePositive,mustBeInteger}
                dynamics (1, 1) struct
                max_queue (:, 1) double = []
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
            if ~isempty(max_queue)                                          % optional slacks for max queues
                assert(length(max_queue) == dynamics.states.w.size(1))
                vars.slack = opti.variable( ...
                                    sum(isfinite(max_queue)), M * Np + 1);
            end
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

            % indices of equality and inequality constraints
            I_g_eq = [];
            I_g_ineq = [];

            % constraints on domains
            ng = opti.ng;
            opti.subject_to(-vars.w(:) + eps <= 0)
            opti.subject_to(-vars.rho(:) + eps <= 0)
            opti.subject_to(-vars.v(:) + eps <= 0)
            if ~isempty(max_queue)
                opti.subject_to(-vars.slack(:) + eps^2 <= 0)
            end
            opti.subject_to(-vars.r(:) + 0.2 <= 0)
            opti.subject_to(vars.r(:) - 1 <= 0)
            I_g_ineq = [I_g_ineq, ng + 1:opti.ng];
            
            % constraints on initial conditions
            ng = opti.ng;
            opti.subject_to(vars.w(:, 1) - pars.w0 == 0)
            opti.subject_to(vars.v(:, 1) - pars.v0 == 0)
            opti.subject_to(vars.rho(:, 1) - pars.rho0 == 0)
            I_g_eq = [I_g_eq, ng + 1:opti.ng];

            % (soft) constraints on queues
            ng = opti.ng;
            if ~isempty(max_queue)
                j = 1;
                for i = 1:length(max_queue)
                    if ~isfinite(max_queue(i))
                        continue
                    end
                    opti.subject_to( ...
                       vars.w(i, :) - vars.slack(j, :) - max_queue(i) <= 0)
                    j = j + 1;
                end
            end
            I_g_ineq = [I_g_ineq, ng + 1:opti.ng];

            % expand control sequence
            r_exp = [repelem(vars.r, 1, M), ...
                repelem(vars.r(:, end), 1, M * (Np - Nc))];

            % constraints on state evolution
            ng = opti.ng;
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
            I_g_eq = [I_g_eq, ng + 1:opti.ng];

            % categorize constraints
            g = opti.g;
            lam_g = opti.lam_g;
            con = struct( ...
                'eq', struct('I', I_g_eq, ...
                             'g', g(I_g_eq), ...
                             'lam_g', lam_g(I_g_eq)), ...
                'ineq', struct('I', I_g_ineq, ...
                               'g', g(I_g_ineq), ...
                               'lam_g', lam_g(I_g_ineq)));

            % save to instance
            obj.Np = Np;
            obj.Nc = Nc;
            obj.M = M;
            obj.dynamics = dynamics;
            obj.max_queue = max_queue;
            obj.opti = opti;
            obj.vars = vars;
            obj.pars = pars;
            obj.con = con;
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

        function [sol, info] = solve(obj, pars, vals, ...
                                    shift_vals, use_fmincon, multistart)
            % SOLVE. Solve the NMPC problem with the given parameter values
            % and initial conditions for the variables.
            arguments
                obj (1, 1) rlmpc.NMPC
                pars (1, 1) struct
                vals (1, 1) struct
                shift_vals (1, 1) logical = true;
                use_fmincon (1, 1) logical = false;
                multistart (1, 1) {mustBePositive,mustBeInteger} = 1
            end

            % if requested, shifts the initial conditions for the MPC by M
            % instants to the left, and pads the right with a simulation
            if shift_vals
                vals = obj.shift_vals(pars, vals);
            end

            % if multistarting, solve the problem n times, returning only 
            % the best solution that is feasible. If none are feasible,
            % pick the best among them.
            if multistart > 1
                for i = 1:multistart % unable to parallelize SWIG references...
                    % perturb initial conditions and solve i-th problem
                    vals_i = vals;
                    vals_i.w = vals.w + randn(size(vals.w)) * (i-1)/5;
                    vals_i.rho = vals.rho + randn(size(vals.v)) * (i-1)/5;
                    vals_i.v = vals.v + randn(size(vals.v)) * (i-1)/5;
                    vals_i.r = vals.r + randn(size(vals.r)) * (i-1)/10;
                    [sol_i, info_i] = obj.solve( ...
                                pars, vals_i, false, use_fmincon);

                    % decide if better than current
                    if (i == 1  || ...                                  % pick the first
                            (~info.success && info_i.success) || ...    % pick first that is feasible
                            ((info.success == info_i.success) ...       % if both (in)feasible, compare fval
                                                    && info_i.f < info.f))
                        sol = sol_i;
                        info = info_i;
                    end
                end
                return
            end

            % enforce feasibility on initial conditions - ipopt will do it
            % again in the restoration phase, but let's help it
            pars.w0   = max(obj.eps, pars.w0);
            pars.rho0 = max(obj.eps, pars.rho0);
            pars.v0   = max(obj.eps, pars.v0);
            vals.w    = max(obj.eps, vals.w);
            vals.rho  = max(obj.eps, vals.rho);
            vals.v    = max(obj.eps, vals.v);
            vals.r    = min(1, max(0.2, vals.r));
            if isfield(vals, 'slack')
                I = find(isfinite(obj.max_queue));
                vals.slack = ...
                        max(obj.eps^2, vals.w(I, :) - obj.max_queue(I));
            end

            % choose which optimizer to run
            if use_fmincon
                [sol, get_value, ~, f, ~, ~, flag, output] = ...
                                    rlmpc.fmc.solveSQP(obj, pars, vals);
                info.f = f;
                info.success = flag > 0;
                if ~info.success
                    msg = split(output.message, '.');
                    info.error = strtrim(msg{1});
                end
                info.get_value = get_value;
            else
                % set parameter values and set initial conditions
                for name = fieldnames(obj.pars)'
                    obj.opti.set_value(obj.pars.(name{1}), pars.(name{1}));
                end
                for name = fieldnames(obj.vars)'
                    obj.opti.set_initial( ...
                                    obj.vars.(name{1}), vals.(name{1}));
                end
    
                % run solver
                info = struct;
                try
                    s = obj.opti.solve();
                    info.success = true;
                    get_value = @(o) s.value(o);
                    info.get_value = get_value;
                catch ME1
                    try
                        stats = obj.opti.debug.stats();
                        info.success = false;
                        info.error = stats.return_status;
                        get_value = @(o) obj.opti.debug.value(o);
                        info.get_value = get_value;
                    catch ME2
                        rethrow(addCause(ME2, ME1))
                    end   
                end
    
                % get outputs
                info.f = get_value(obj.opti.f);
                sol = struct;
                for name = fieldnames(obj.vars)'
                    sol.(name{1}) = get_value(obj.vars.(name{1}));
                end
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

        function vals = shift_vals(obj, pars, vals)
            arguments
                obj (1, 1) rlmpc.NMPC
                pars (1, 1) struct
                vals (1, 1) struct
            end
            % shift the to left, de facto forwarding M instants in time
            vals.w = [vals.w(:, obj.M + 1:end), ...
                                    nan(size(vals.w, 1), obj.M)];
            vals.rho = [vals.rho(:, obj.M + 1:end), ...
                                    nan(size(vals.rho, 1), obj.M)];
            vals.v = [vals.v(:, obj.M + 1:end), ...
                                    nan(size(vals.v, 1), obj.M)];

            % draw random last action
            r_rand = min(1, max(0.2, ...
                    mean(vals.r, 2) + randn(size(vals.r, 1), 1) * 0.1));
            vals.r = [vals.r(:, 2:end), r_rand];
            
            % pad to the erased instants to the right with a simulation
            if isfield(pars, 'pars_Veq_approx')
                pars_dyn = {pars.rho_crit, pars.pars_Veq_approx};
            else
                pars_dyn = {pars.rho_crit, pars.a, pars.v_free};
            end
            for k = obj.M * (obj.Np - 1) + 1:obj.M * obj.Np
                [~, w_next, ~, rho_next, v_next] = obj.dynamics.f( ...
                            vals.w(:, k), vals.rho(:, k), vals.v(:, k), ...
                            vals.r(:, end), pars.d(:, k), pars_dyn{:});
                vals.w(:, k + 1) = full(w_next);
                vals.rho(:, k + 1) = full(rho_next);
                vals.v(:, k + 1) = full(v_next);
            end

            % slacks are taken care when feasibility is enforced
        end
    end
end
