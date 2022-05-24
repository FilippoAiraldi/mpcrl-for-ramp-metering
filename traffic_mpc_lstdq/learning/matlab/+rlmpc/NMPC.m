classdef NMPC < handle
    % NMPC. Wrapper around casadi.NLPSolver to facilitate solving MPC 
    % problem for the given 3-link metanet problem.
    


    properties (GetAccess = public, SetAccess = protected)
        %
        name (1, :) char
        M, Nc, Np (1, 1) double
        %
        vars (1, 1) struct = struct;
        pars (1, 1) struct = struct;
        %
        p (:, 1) = []
        %
        x (:, 1) = []
        lam_lbx (:, 1) = []
        lam_ubx (:, 1) = []
        lbx (:, 1) = []
        ubx (:, 1) = []
        %
        g (:, 1) = []
        lam_g (:, 1) = []
        lbg (:, 1) = []
        ubg (:, 1) = []
        Ig_eq (:, 1) = []
        Ig_ineq (:, 1) = []
        %
        f (1, 1)
        %
        solver (1, 1)
        opts (1, 1)
        %
        dynamics (1, 1) struct
    end


    
    methods (Access = public)
        function obj = NMPC(name, Np, Nc, M, dyn, max_queue, soft_con, ...
                                    eps, flow_as_control, rho_max, C, T)
            % NMPC. Builds an instance of an NMPC with the corresponding
            % horizons and dynamics.
            arguments
                name (1, :) char {mustBeTextScalar}
                Np, Nc, M (1, 1) double {mustBePositive,mustBeInteger}
                dyn (1, 1) struct
                max_queue (:, 1) double = []
                soft_con (1, 1) logical = false
                eps (1, 1) double {mustBeNonnegative} = 0
                flow_as_control (1, 1) logical = false
                rho_max (1, 1) double = 0   % only needed if flow_as_control=true
                C (:, 1) double = 0         % only needed if flow_as_control=true
                T (1, 1) double = 0         % only needed if flow_as_control=true
            end
            obj.name = name;

            % create variables
            if ~soft_con
                bnd = eps;
            else
                bnd = -inf;
            end
            w = obj.add_var('w', [dyn.states.w.size(1), M * Np + 1], bnd);
            rho = obj.add_var('rho', [dyn.states.v.size(1), M*Np+1], bnd);
            v = obj.add_var('v', [dyn.states.v.size(1), M * Np + 1], bnd);
            if soft_con
                slack_w = obj.add_var('slack_w', size(w), eps^2);
                slack_rho = obj.add_var('slack_rho', size(rho), eps^2);
                slack_v = obj.add_var('slack_v', size(v), eps^2);
            end
            if ~isempty(max_queue) % optional slacks for max queues
                assert(length(max_queue) == dyn.states.w.size(1))
                slack_w_max = obj.add_var('slack_w_max', ...
                            [sum(isfinite(max_queue)), M * Np + 1], eps^2);
            end
            
            % based on what is the control action, bounds differ
            if ~flow_as_control
                bnd = {0.2, 1}; % metering rate is between [0.2, 1]
            else
                if isequal([size(dyn.input.r, 1), length(C)], [1, 1]) % origin is not a ramp, and is not controlled
                    bnd = {2e2, C(1)};
                elseif isequal([size(dyn.input.r, 1), length(C)], [1, 2]) % origin is a ramp, but is not controlled
                    bnd = {2e2, C(2)};
                else % [2, 2] % origin is a ramp, and is controlled
                    bnd = {2e2, C};
                end 
                % this fixed lb might cause infeasibility when too congested
                
            end
            r = obj.add_var('r', [size(dyn.input.r, 1), Nc], bnd{:});

			% create parameters
            d = obj.add_par('d', [dyn.dist.d.size(1), M * Np]);
            w0 = obj.add_par('w0', [dyn.states.w.size(1), 1]);
            rho0 = obj.add_par('rho0', [dyn.states.rho.size(1), 1]);
            v0 = obj.add_par('v0', [dyn.states.v.size(1), 1]);

            % create params for the system dynamics - might vary
            pars_dyn = {}; % will be helpful later to call dynamics.f
            for par = fieldnames(dyn.pars)'
                pars_dyn{end + 1} = obj.add_par( ...
                                par{1}, size(dyn.pars.(par{1}))); %#ok<AGROW> 
            end

            % if constraints on domains are soft, then we must specify them
            % manually
            if soft_con
                obj.add_con('w_pos', -w - slack_w + eps, -inf, 0)
                obj.add_con('rho_pos', -rho - slack_rho + eps, -inf, 0)
                obj.add_con('v_pos', -v - slack_v + eps, -inf, 0)
            end

            % (soft) constraints on queues
            if ~isempty(max_queue)
                I = find(isfinite(max_queue));
                for i = 1:size(slack_w_max, 1)
                    obj.add_con('w_max', ...
                      w(I(i), :) - slack_w_max(i, :) - max_queue(I(i)), ...
                      -inf, 0);
                end
            end

            % constraints on initial conditions
            obj.add_con('w_init', w(:, 1) - w0, 0, 0)
            obj.add_con('rho_init', rho(:, 1) - rho0, 0, 0)
            obj.add_con('v_init', v(:, 1) - v0, 0, 0)

            % expand constraint to match size of states
            r_exp = [repelem(r, 1, M), repelem(r(:, end), 1, M * (Np-Nc))];
            
            % if flow is control action, add constraints to its value
            % mimicking the min term in the dynamics
            if flow_as_control
                assert(rho_max > 0 && T > 0 && all(C > 0))
                if isequal([size(r, 1), size(w, 1)], [1, 1]) % origin is not a ramp, and is not controlled
                    I = {1, 2, 1};
                elseif isequal([size(r, 1), size(w, 1)], [1, 2]) % origin is a ramp, but is not controlled
                    I = {1, 2, 2};
                else % [2, 2] % origin is a ramp, and is not controlled
                    I = {1:2, 1:2, 1:2};
                end
                obj.add_con('flow_control_min1', ...
                    r_exp(I{1}, :) - d(I{2}, :) - w(I{3}, 1:end-1) / T, ...
                    -inf, 0)

                if isequal([size(r, 1), length(C)], [1, 1]) % origin is not a ramp, and is not controlled
                    I = {1, 1, 3};
                elseif isequal([size(r, 1), length(C)], [1, 2]) % origin is a ramp, but is not controlled
                    I = {1, 2, 3};
                else % [2, 2] % origin is a ramp, and is controlled
                    I = {1:2, 1:2, [1, 3]};
                end 
                obj.add_con('flow_control_min2', ...
                    (rho_max - obj.pars.rho_crit) * r_exp(I{1}, :) - ...
                             C(I{2}) .* (rho_max - rho(I{3}, 1:end-1)), ...
                    -inf, 0);
            end

            % constraints on state evolution
            for k = 1:M * Np
                [~, w_next, ~, rho_next, v_next] = dyn.f(...
                    w(:, k), ...
                    rho(:, k), ...
                    v(:, k), ...
                    r_exp(:, k), ...
                    d(:, k), ...
                    pars_dyn{:});
                k_ = num2str(k);
                obj.add_con(['w_', k_],   w(:, k + 1)   - w_next, 0, 0)
                obj.add_con(['rho_', k_], rho(:, k + 1) - rho_next, 0, 0)
                obj.add_con(['v_', k_],   v(:, k + 1)   - v_next, 0, 0)
            end

            % save stuff
            obj.M = M;
            obj.Nc = Nc;
            obj.Np = Np;
            obj.dynamics = dyn;
        end

        function par = add_par(obj, name, dims)
            % ADD_PAR. Adds a parameter to the NMPC instance, with the
            % given name and size.
            arguments
                obj (1, 1) rlmpc.NMPC
                name (1, :) char {mustBeTextScalar}
                dims (1, 2) double {mustBePositive,mustBeInteger}
            end
            assert(~any(strcmp(fieldnames(obj.pars), name), 'all'), ...
                'parameter name already in use')
            
            par = casadi.SX.sym(name, dims);
            obj.p = vertcat(obj.p, par(:));
            obj.pars.(name) = par;
        end

        function var = add_var(obj, name, dims, lb, ub)
            % ADD_VAR. Adds a variable to the NMPC instance, with the
            % given name and size.
            arguments
                obj (1, 1) rlmpc.NMPC 
                name (1, :) char {mustBeTextScalar}
                dims (1, 2) double {mustBePositive,mustBeInteger}
                lb (:, :) double = -inf
                ub (:, :) double = inf
            end
            assert(~any(strcmp(fieldnames(obj.pars), name), 'all'), ...
                'variable name already in use')

            expansion = dims ./ size(lb);
            lb = repmat(lb, expansion(1), expansion(2));
            expansion = dims ./ size(ub);
            ub = repmat(ub, expansion(1), expansion(2));

            assert(isequal(size(lb), dims) && isequal(size(ub), dims), ...
                'dimensions are incompatible')
            assert(all(lb <= ub, 'all'), 'invalid bounds');

            var = casadi.SX.sym(name, dims);
            obj.x = vertcat(obj.x, var(:));
            obj.lbx = vertcat(obj.lbx, lb(:));
            obj.ubx = vertcat(obj.ubx, ub(:));
            obj.vars.(name) = var;

            % create also the multiplier associated to the variable
            % to understand what this means - I think the dual solution
            mul = casadi.SX.sym(append('lam_', name, 'lb'), dims);
            obj.lam_lbx = vertcat(obj.lam_lbx, mul(:));
            mul = casadi.SX.sym(append('lam_', name, 'ub'), dims);
            obj.lam_ubx = vertcat(obj.lam_ubx, mul(:));
        end

        function add_con(obj, name, g, lb, ub)
            % ADD_CON. Adds a constraint to the NMPC instance. If bounds 
            % are equal, then it is an equality; otherwise inequality.
            arguments
                obj (1, 1) rlmpc.NMPC 
                name (1, :) char {mustBeTextScalar}
                g (:, :) casadi.SX
                lb (:, :) double
                ub (:, :) double
            end
            dims = size(g);
            if ~isvector(g)
                warning('first time using matrices, check everything')
            end

            expansion = dims ./ size(lb);
            lb = repmat(lb, expansion(1), expansion(2));
            expansion = dims ./ size(ub);
            ub = repmat(ub, expansion(1), expansion(2));

            assert(isequal(size(lb), dims) && isequal(size(ub), dims), ...
                'dimensions are incompatible')
            assert(all(lb <= ub, 'all'), 'invalid bounds');

            obj.g = vertcat(obj.g, g(:));
            obj.lbg = vertcat(obj.lbg, lb(:));
            obj.ubg = vertcat(obj.ubg, ub(:));
            ng = obj.ng;
            L = numel(g);
            if all(lb == ub, 'all')
                % equality
                obj.Ig_eq = vertcat(obj.Ig_eq, (ng - L + 1:ng)');
            elseif all(lb < ub, 'all')
                % inequality
                obj.Ig_ineq = vertcat(obj.Ig_ineq , (ng - L + 1:ng)');
            else
                error('cannot categorize mixed constraints')
            end

            % create also the multiplier associated to the constraint
            mul = casadi.SX.sym(append('lam_g_', name), [L, 1]);
            obj.lam_g = vertcat(obj.lam_g, mul(:));
        end

        function minimize(obj, objective)
            % MINIMIZE. Sets the objective to minimize.
            arguments
                obj (1, 1) rlmpc.NMPC 
                objective (1, 1) casadi.SX
            end
            obj.f = objective;
        end

        function init_solver(obj, opts)
            % INIT_SOLVER. Initializes the solver, either ipopt or fmincon.
            arguments
                obj (1, 1) rlmpc.NMPC 
                opts (1, 1) 
            end

            if isa(opts, 'optim.options.Fmincon')
                % do this only once
                file = sprintf('F_gen_%s.c', obj.name);
                if ~isfile(file)
                    warning( ...
                        ['generating mex file for SPQ; delete "', ...
                          file, '" to force repeating the process.'])
        
                    % compute symbolic derivatives and generate code
                    df = jacobian(obj.f, obj.x)';
                    dg_eq = jacobian(obj.g(obj.Ig_eq), obj.x)';
                    dg_ineq = jacobian(obj.g(obj.Ig_ineq), obj.x)';
                    F = casadi.Function('F', {obj.p, obj.x}, ...
                             {obj.f, df, obj.g(obj.Ig_eq), ...
                                    dg_eq, obj.g(obj.Ig_ineq), dg_ineq});
                    F.generate(file, struct('mex', true));
        
                    % load it as a mex
                    mex(file, '-largeArrayDims');
                end
            else
                % create solver
                if isfield(opts, 'ipopt')
                    solver_type = 'ipopt';
                else
                    solver_type = 'sqpmethod';
                end
                nlp = struct('x',obj.x, 'p',obj.p, 'g', obj.g, 'f', obj.f);
                obj.solver = casadi.nlpsol( ...
                    ['solver_', obj.name], solver_type, nlp, opts);
            end
            obj.opts = opts;
        end

        function i = nx(obj)
            i = length(obj.x);
        end

        function i = ng(obj)
            i = length(obj.g);
        end

        function i = np(obj)
            i = length(obj.p);
        end

        function L = lagrangian(obj)
            L = obj.f ...
                + obj.lam_g' * obj.g ...
                + obj.lam_lbx' * (obj.lbx - obj.x) ...
                + obj.lam_ubx' * (obj.x - obj.ubx);
        end

        function v = concat_pars(obj, names)
            % CONCAT_PARS. Concatenates in a single vertical array the NMPC 
            % parameters whose name appears in the list.
            arguments
                obj (1, 1) rlmpc.NMPC
                names (:, 1) cell
            end
            v = cellfun(@(n) obj.pars.(n), names, 'UniformOutput', false);
            v = vertcat(v{:});
        end

        function [sol, info] = solve(obj,pars,vals, shift_vals, multistart)
            % SOLVE. Solve the NMPC problem with the given parameter values
            % and initial conditions for the variables.
            arguments
                obj (1, 1) rlmpc.NMPC
                pars (1, 1) struct
                vals (1, 1) struct
                shift_vals (1, 1) logical = true;
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
            vals = orderfields(vals, obj.vars);
            pars = orderfields(pars, obj.pars);

            % decide which algorithm to use
            if isa(obj.opts, 'optim.options.Fmincon')
                [sol, info] = obj.solve_fmincon_multistart( ...
                                                pars, vals, multistart);
            else
                [sol, info] = obj.solve_nlp_multistart( ...
                                                pars, vals, multistart);
            end
        end
    end



    methods (Access = protected)
        function [sol, info] = solve_fmincon_multistart(obj, pars, ...
                                                        vals, multistart)
            % pre-compute stuff to avoid obj in the parfor loop
            p_ = obj.subsevalf(obj.p, obj.pars, pars);
            lbx_ = obj.lbx;
            ubx_ = obj.ubx;
            varnames = fieldnames(obj.vars)';
            name_ = obj.name;
            opts_ = obj.opts;

            % call SQP solver 
            if multistart == 1
                x0 = obj.subsevalf(obj.x, obj.vars, vals);
                x0 = max(lbx_, min(ubx_, x0));
                sol = rlmpc.solveSQP(name_, p_, x0, lbx_, ubx_, opts_);
            else
                sols = cell(1, multistart);
                parfor i = 1:multistart % multistart / 2)
                    vals_i = rlmpc.NMPC.perturb_vals(vals, i);
                    x0 = cellfun(@(n) vals_i.(n)(:), varnames, ...
                                                'UniformOutput', false);
                    x0 = vertcat(x0{:});
                    x0 = max(lbx_, min(ubx_, x0));
                    sols{i} = rlmpc.solveSQP(name_, p_, ...
                                                    x0, lbx_, ubx_, opts_);
                end
    
                % find best among all solutions
                sol = sols{1};
                i_opt = 1; 
                for i = 2:multistart
                    sol_i = sols{i};
                    if (~sol.success && sol_i.success) || ...               % pick first that is feasible
                       ((sol.success == sol_i.success) && sol_i.f < sol.f)  % if both (in)feasible, compare f 
                                                            
                        sol = sol_i;
                        i_opt = i;
                    end
                end
            end

            % put multiplier in a unique vector
%             assert(all(sol.lambda.ineqnonlin >= 0, 'all') && ...
%                 all(sol.lambda.lower >= 0, 'all') && ...
%                 all(sol.lambda.upper >= 0, 'all'), 'invalid multipliers')
            lam_g_ = nan(obj.ng, 1);
            lam_g_(obj.Ig_eq) = sol.lambda.eqnonlin;
            lam_g_(obj.Ig_ineq) = sol.lambda.ineqnonlin;

            % build info
            S = [obj.p; obj.x; obj.lam_g; obj.lam_lbx; obj.lam_ubx];
            D = [p_; sol.x; lam_g_; sol.lambda.lower; sol.lambda.upper];
            get_value = @(o) rlmpc.NMPC.subsevalf(o, S, D);
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
    
        function [sol, info] = solve_nlp_multistart(obj, pars, vals, ...
                                                                multistart)
            % pre-compute stuff to avoid obj in the parfor loop
            slvr = obj.solver;
            p_ = obj.subsevalf(obj.p, obj.pars, pars);
            lbx_ = obj.lbx;
            ubx_ = obj.ubx;
            lbg_ = obj.lbg;
            ubg_ = obj.ubg;
            varnames = fieldnames(obj.vars)';

            % call SQP solver 
            if multistart == 1
                x0 = obj.subsevalf(obj.x, obj.vars, vals);
                x0 = max(lbx_, min(ubx_, x0));
                sol = slvr('x0', x0, 'p', p_, 'lbx', lbx_, ...
                          'ubx', ubx_, 'lbg', lbg_, 'ubg', ubg_); 
                status = slvr.stats.return_status;
                success = strcmp(status, 'Solve_Succeeded');
            else
                sols = cell(1, multistart);
                statuses = cell(1, multistart);
                parfor i = 1:multistart % multistart / 2)
                    vals_i = rlmpc.NMPC.perturb_vals(vals, i);
                    x0 = cellfun(@(n) vals_i.(n)(:), varnames, ...
                                                'UniformOutput', false);
                    x0 = vertcat(x0{:});
                    x0 = max(lbx_, min(ubx_, x0));
                    sol = slvr('x0', x0, 'p', p_, 'lbx', lbx_, ...
                                   'ubx', ubx_, 'lbg', lbg_, 'ubg', ubg_); %#ok<PFBNS> 
                    sol.f = full(sol.f);
                    sols{i} = sol;
                    statuses{i} = slvr.stats.return_status;
                end

                % find best among all solutions
                sol = sols{1};
                success = strcmp(statuses{1}, 'Solve_Succeeded');
                i_opt = 1; 
                for i = 2:multistart
                    sol_i = sols{i};
                    success_i = strcmp(statuses{i}, 'Solve_Succeeded');
                    if (~success && success_i) || ...               % pick first that is feasible
                       ((success == success_i) && sol_i.f < sol.f)  % if both (in)feasible, compare f 
                                                            
                        sol = sol_i;
                        success = success_i; 
                        i_opt = i;
                    end
                end
                status = statuses{i_opt};
            end

            % from the unique lam_x, extract lam_lbx and lam_ubx
            lam_lbx_ = -min(0, sol.lam_x);
            lam_ubx_ =  max(0, sol.lam_x);

            % build info
            S = [obj.p; obj.x; obj.lam_g; obj.lam_lbx; obj.lam_ubx];
            D = [p_; sol.x; sol.lam_g; lam_lbx_; lam_ubx_];
            get_value = @(o) rlmpc.NMPC.subsevalf(o, S, D);
            info = struct('f', full(sol.f), 'msg', status, ...
                          'success', success, 'get_value', get_value);  
            if multistart ~= 1
                info.i_opt = i_opt;
            end
        
            % compute per-variable solution
            sol = struct;
            for n = varnames
                sol.(n{1}) = get_value(obj.vars.(n{1}));
            end
        end
    end



    methods (Access = protected, Static)
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

        function vals = perturb_vals(vals, mag)
            % perturb initial conditions by some magnitude
            b = (mag - 1) / 20;
            vals.w = vals.w + randn(size(vals.w)) * b;
            vals.rho = vals.rho + randn(size(vals.v)) * b;
            vals.v = vals.v + randn(size(vals.v)) * b;
            vals.r = vals.r + randn(size(vals.r)) * b / 2;
        end
    
        function y = subsevalf(expr, old, new, eval)
            % SUBSEVAL. Substitute in the expression the old variable with
            % the new, and evaluate the expression if required.
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
            if nargin < 4 || eval
                y = full(evalf(y));
            end
        end
    end
end
