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
        con (1, 1) struct
        opts (1, 1) struct
    end



    methods (Access = public) 
        function obj = NMPC(Np, Nc, M, dynamics, opts, ...
                                        max_queue, soft_domain_con, eps)
            % NMPC. Builds an instance of an NMPC with the corresponding
            % horizons and dynamics.
            arguments
                Np, Nc, M (1, 1) double {mustBePositive,mustBeInteger}
                dynamics (1, 1) struct
                opts (1, 1) struct = struct
                max_queue (:, 1) double = []
                soft_domain_con (1, 1) logical = false
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
            if soft_domain_con
                vars.slack_w = opti.variable( ...
                                    vars.w.size(1), vars.w.size(2));
                vars.slack_rho = opti.variable( ...
                                    vars.rho.size(1), vars.rho.size(2));
                vars.slack_v = opti.variable( ...
                                    vars.v.size(1), vars.v.size(2));
            end
            if ~isempty(max_queue)                                          % optional slacks for max queues
                assert(length(max_queue) == dynamics.states.w.size(1))
                vars.slack_max_w = opti.variable( ...
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

            % constraints on domains
            for n = fieldnames(vars)'
                if startsWith(n{1}, 'slack')
                    opti.subject_to(-vars.(n{1})(:) + eps^2 <= 0)
                elseif strcmp(n{1}, 'r')
                    opti.subject_to(-vars.r(:) + 0.2 <= 0)
                    opti.subject_to(vars.r(:) - 1 <= 0)
                else
                    if ~soft_domain_con
                        opti.subject_to(-vars.(n{1})(:) + eps <= 0)
                    else
                        opti.subject_to(-vars.(n{1})(:) - ...
                                    vars.(['slack_', n{1}])(:) + eps <= 0)
                    end
                end
            end

            % (soft) constraints on queues
            if ~isempty(max_queue)
                I = find(isfinite(max_queue));
                for i = 1:size(vars.slack_max_w, 1)
                    opti.subject_to(vars.w(I(i), :) - ...
                            vars.slack_max_w(i, :) - max_queue(I(i)) <= 0)
                end
            end

            % constraints on initial conditions
            opti.subject_to(vars.w(:, 1) - pars.w0 == 0)
            opti.subject_to(vars.v(:, 1) - pars.v0 == 0)
            opti.subject_to(vars.rho(:, 1) - pars.rho0 == 0)

            % constraints on state evolution
            r_exp = [repelem(vars.r, 1, M), ...
                                repelem(vars.r(:, end), 1, M * (Np - Nc))];
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

            % categorize constraints
            mask_eq = logical(full(evalf(opti.lbg == opti.ubg)));
            Ieq = find(mask_eq);
            Iin = find(~mask_eq);
            g = opti.g;
            lam_g = opti.lam_g;
            con = struct;
            con.eq = struct('I', Ieq, 'g', g(Ieq), 'lam_g', lam_g(Ieq));
            con.ineq = struct('I', Iin, 'g', g(Iin), 'lam_g', lam_g(Iin));
            
            % set the solver
            opti.solver('ipopt', opts.plugin, opts.solver);

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
            obj.opts = opts;
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
            % instants to the left, and pads the right with a simulation.
            % This step is independent of the algorithm and multistart.
            if shift_vals
                vals = obj.shift_vals( ...
                                pars, vals, obj.M, obj.Np, obj.dynamics.f);
            end

            % order pars and vars according to the order of creation
            % vals = orderfields(vals, obj.vars);
            % pars = orderfields(pars, obj.pars);

            % decide which algorithm to use
            if use_fmincon
                [sol, info] = obj.solve_fmincon_multistart( ...
                                                pars, vals, multistart);
            else
                [sol, info] = obj.solve_opti_multistart( ...
                                                pars, vals, multistart);
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



    methods (Access = protected)
        function [sol, info] = solve_fmincon_multistart(obj, pars, ...
                                                        vals, multistart)
            % solve_fmincon must be obj-agnostic in order to run in
            % parallel. So here we prepare its inputs.

            % do this only once
            if ~isfile('F_gen.c')
                warning(['generating mex file for SPQ solver; delete ' ...
                            '"F_gen.c" to force repeating the process.'])

                % compute symbolic derivatives and generate code
                df = jacobian(obj.opti.f, obj.opti.x)';
                dg_eq = jacobian(obj.con.eq.g, obj.opti.x)';
                dg_ineq = jacobian(obj.con.ineq.g, obj.opti.x)';
                F = casadi.Function('F', {obj.opti.p, obj.opti.x}, ...
                                 {obj.opti.f, df, obj.con.eq.g, ...
                                        dg_eq, obj.con.ineq.g, dg_ineq});
                F.generate('F_gen.c', struct('mex', true));

                % load it as a mex
                mex('F_gen.c', '-largeArrayDims');
            end

            % save some variables, so obj does not enter the loop
            eps_ = obj.eps;
            max_queue_ = obj.max_queue;
            opts_ = obj.opts.fmincon;
            varnames = fieldnames(obj.vars)';
            parnames = fieldnames(obj.pars)';

            % call SQP solver 
            if multistart == 1
                sol = rlmpc.solveSQP( ...
                                    pars, vals, varnames, parnames, opts_);
            else
                % solve in parallel
                sols = cell(1, multistart);
                parfor i = 1:multistart % multistart / 2)
                    % perturb initial conditions
                    vals_i = rlmpc.NMPC.perturb_vals(vals, i);
                    [pars_i, vals_i] = rlmpc.NMPC.enforce_feasibility( ...
                                pars, vals_i, eps_, max_queue_);
    
                    % Solve the SQP
                    sols{i} = rlmpc.solveSQP( ...
                                pars_i, vals_i, varnames, parnames, opts_);
                end
    
                % find best among all solutions
                sol = sols{1};
                i_opt = 1; 
                for i = 2:multistart
                    sol_i = sols{i};
                    if (~sol_i.success && sol_i.success) || ...             % pick first that is feasible
                       ((sol_i.success == sol_i.success) && ...             % if both (in)feasible, compare f 
                                                        sol_i.f < sol_i.f)    
                        sol = sol_i;
                        i_opt = i;
                    end
                end
            end

            % put multiplier in a unique vector
            lam_g = nan(obj.opti.ng, 1);
            lam_g(obj.con.eq.I) = sol.lam_g.eqnonlin;
            lam_g(obj.con.ineq.I) = sol.lam_g.ineqnonlin;
            
            % build info
            get_value = @(o) rlmpc.NMPC.subsevalf(o, ...
                            [obj.opti.p; obj.opti.x; obj.opti.lam_g], ...
                            [sol.p; sol.x; lam_g]);
            info = struct('f', sol.f, 'success', sol.success, ...
                                'msg', sol.msg, 'get_value', get_value);
            if multistart ~= 1
                info.i_opt = i_opt;
            end
        
            % compute per-variable solution
            sol = struct;
            for n = varnames
                sol.(n{1}) = get_value(obj.vars.(n{1}));
            end
        end

        function [sol, info] = solve_opti(obj, pars, vals)
            % set parameter values and set initial conditions
            for name = fieldnames(obj.pars)'
                obj.opti.set_value(obj.pars.(name{1}), pars.(name{1}));
            end
            for name = fieldnames(obj.vars)'
                obj.opti.set_initial(obj.vars.(name{1}), vals.(name{1}));
            end
    
            % run solver
            info = struct;
            try
                s = obj.opti.solve();
                info.success = true;
                info.get_value = @(o) s.value(o);
            catch ME1
                try
                    info.success = false;
                    info.get_value = @(o) obj.opti.debug.value(o);
                catch ME2
                    rethrow(addCause(ME2, ME1))
                end   
            end
            
            % get outputs
            info.f = info.get_value(obj.opti.f);
            info.msg = obj.opti.debug.stats.return_status;
            sol = struct;
            for name = fieldnames(obj.vars)'
                sol.(name{1}) = info.get_value(obj.vars.(name{1}));
            end
        end

        function [sol, info] = solve_opti_multistart(obj, pars, vals, ...
                                                                multistart)
            % check if problem can be solved just once
            if multistart == 1
                [sol, info] = solve_opti(obj, pars, vals);
                return
            end

            % to run the problem in parallel, we have to convert it to an
            % NLP (see https://web.casadi.org/blog/parfor/)
            nlp = struct('x', obj.opti.x, 'p', obj.opti.p, ...
                         'f', obj.opti.f, 'g', obj.opti.g);
            opts_ = obj.opts.plugin;
            opts_.ipopt = obj.opts.solver;
            solver = casadi.nlpsol('solver', 'ipopt', nlp, opts_);

            % save some variables, so obj does not enter the loop
            eps_ = obj.eps;
            max_queue_ = obj.max_queue;
            lbg_ = evalf(obj.opti.lbg); % for some reason, need to copy these
            ubg_ = evalf(obj.opti.ubg);
            varnames = fieldnames(obj.vars)';
            parnames = fieldnames(obj.pars)';
            
            % solve in parallel
            sols = cell(1, multistart);
            infos = cell(1, multistart);
            parfor i = 1:multistart % multistart / 2)
                % perturb initial conditions
                vals_i = rlmpc.NMPC.perturb_vals(vals, i);
                [pars_i, vals_i] = rlmpc.NMPC.enforce_feasibility( ...
                            pars, vals_i, eps_, max_queue_);

                % convert to vectors
                x0 = cellfun(@(n) vals_i.(n)(:), varnames, ...
                                                'UniformOutput', false);
                x0 = vertcat(x0{:});
                p = cellfun(@(n) pars_i.(n)(:), parnames, ...
                                                'UniformOutput', false);
                p = vertcat(p{:});

                % Solve the NLP
                sol = solver('x0' , x0, 'p', p, ...
                                 'lbx', 0, 'ubx', inf, ...
                                 'lbg', lbg_, 'ubg', ubg_); %#ok<PFBNS> 
                f = full(sol.f);
                sols{i} = struct('f', f, 'x', full(sol.x), 'p', p, ...
                                 'g', full(sol.g), ...
                                 'lam_g', full(sol.lam_g));
                stats = solver.stats;
                
                % build info 
                info = struct;  
                info.f = f;
                if strcmp(stats.return_status, 'Solve_Succeeded')
                    info.success = true;
                else
                    info.success = false;
                    info.error = stats.return_status;
                end
                infos{i} = info;
            end

            % find best among all solutions
            sol = sols{1};
            info = infos{1};
            i_opt = 1;
            for i = 2:multistart
                sol_i = sols{i};
                info_i = infos{i};
                if (~info.success && info_i.success) || ...                 % pick first that is feasible
                   ((info.success == info_i.success) && info_i.f < info.f)  % if both (in)feasible, compare f   
                    sol = sol_i;
                    info = info_i;
                    i_opt = i;
                end
            end
            info.i_opt = i_opt;

            % build a function to get the values 
            info.get_value = @(o) rlmpc.NMPC.subsevalf(o, ...
                    [obj.opti.p; obj.opti.x; obj.opti.lam_g], ...
                    [sol.p; sol.x; sol.lam_g]);
        
            % compute per-variable solution
            sol = struct;
            for n = varnames
                sol.(n{1}) = info.get_value(obj.vars.(n{1}));
            end
        end
    end



    methods (Access = protected, Static)
        function vals = perturb_vals(vals, mag)
            % perturb initial conditions by some magnitude
            b = (mag - 1) / 5;
            vals.w = vals.w + randn(size(vals.w)) * b;
            vals.rho = vals.rho + randn(size(vals.v)) * b;
            vals.v = vals.v + randn(size(vals.v)) * b;
            vals.r = vals.r + randn(size(vals.r)) * b / 2;
        end

        function vals = shift_vals(pars, vals, M, Np, Fdyn)
            % shift to left, de facto forwarding M instants in time
            vals.w = [vals.w(:, M + 1:end), nan(size(vals.w, 1), M)];
            vals.rho = [vals.rho(:, M + 1:end), nan(size(vals.rho, 1), M)];
            vals.v = [vals.v(:, M + 1:end), nan(size(vals.v, 1), M)];
        
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
            for k = M * (Np - 1) + 1:M * Np
                [~, w_next, ~, rho_next, v_next] = Fdyn( ...
                            vals.w(:, k), vals.rho(:, k), vals.v(:, k), ...
                            vals.r(:, end), pars.d(:, k), pars_dyn{:});
                vals.w(:, k + 1) = full(w_next);
                vals.rho(:, k + 1) = full(rho_next);
                vals.v(:, k + 1) = full(v_next);
            end
        
            % slacks are taken care when feasibility is enforced
        end

        function [pars, vals] = enforce_feasibility( ...
                                                pars, vals, eps, max_queue)
            % enforce feasibility on initial conditions - ipopt will do it
            % again in the restoration phase, but let's help it
            pars.w0   = max(eps, pars.w0);
            pars.rho0 = max(eps, pars.rho0);
            pars.v0   = max(eps, pars.v0);
            vals.w    = max(eps, vals.w);
            vals.rho  = max(eps, vals.rho);
            vals.v    = max(eps, vals.v);
            vals.r    = min(1, max(0.2, vals.r));
            if isfield(vals, 'slack_max_w')
                I = find(isfinite(max_queue));
                vals.slack_max_w = ...
                        max(eps^2, vals.w(I, :) - max_queue(I));
            end
        end
    
        function y = subsevalf(expr, old, new)
            y = expr;
            if isstruct(old)
                assert(isstruct(new))
                for n = fieldnames(old)'
                    y = casadi.substitute(y, old.(n{1}), new.(n{1}));
                end
            elseif iscell(old)
                assert(iscell(new))
                for n = 1:length(cell)
                    y = casadi.substitute(y, old{n}, new{n});
                end
            else
                y = casadi.substitute(y, old, new);
            end
            y = full(evalf(y));
        end
    end
end
