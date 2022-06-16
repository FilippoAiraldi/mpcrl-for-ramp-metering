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
        function obj = NMPC(name, model, sim, mpc, dyn)
            % NMPC. Builds an instance of an NMPC with the corresponding
            % horizons and dynamics.
            arguments
                name (1, :) char {mustBeTextScalar}
                model (1, 1) struct
                sim (1, 1) struct
                mpc (1, 1) struct
                dyn (1, 1) struct
            end
            obj.name = name;

            max_queue = model.max_queue;
            rho_max = model.rho_max;
            C = model.C;
            T = sim.T;
            Np = mpc.Np;
            Nc = mpc.Nc;
            M = mpc.M;

            % create state variables
            w = obj.add_var('w', [dyn.states.w.size(1), M * Np + 1], 0);
            rho = obj.add_var('rho', [dyn.states.v.size(1), M*Np+1], 0);
            v = obj.add_var('v', [dyn.states.v.size(1), M * Np + 1], 0);
            slack_w_max = obj.add_var('slack_w_max', [1, M * Np + 1], 0);
            
            % create control action (flow of ramp O2)
            r = obj.add_var('r', [size(dyn.input.r, 1), Nc], 2e2, C(2));

			% create parameters
            d = obj.add_par('d', [dyn.dist.d.size(1), M * Np]);
            w0 = obj.add_par('w0', [dyn.states.w.size(1), 1]);
            rho0 = obj.add_par('rho0', [dyn.states.rho.size(1), 1]);
            v0 = obj.add_par('v0', [dyn.states.v.size(1), 1]);
            pars_dyn = {}; % will be helpful later to call dynamics.f
            for par = fieldnames(dyn.pars)'
                pars_dyn{end + 1} = obj.add_par( ...
                                par{1}, size(dyn.pars.(par{1}))); %#ok<AGROW> 
            end

            % (soft) constraints on queues
            obj.add_con('w_max', w(2, :) - slack_w_max - max_queue,-inf,0);

            % constraints on initial conditions
            obj.add_con('w_init', w(:, 1) - w0, 0, 0)
            obj.add_con('rho_init', rho(:, 1) - rho0, 0, 0)
            obj.add_con('v_init', v(:, 1) - v0, 0, 0)

            % expand constraint to match size of states
            r_exp = [repelem(r, 1, M), repelem(r(:, end), 1, M * (Np-Nc))];
            
            % Since flow is control action, add constraints to its value
            % mimicking the min term in the dynamics
            obj.add_con('flow_control_min1', ...
                r_exp - d(2, :) - w(2, 1:end-1) / T, -inf, 0)
            obj.add_con('flow_control_min2', ...
                (rho_max - obj.pars.rho_crit) * r_exp - ...
                         C(2) .* (rho_max - rho(3, 1:end-1)), -inf, 0);

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
                opts (1, 1) struct
            end

            % create solver
            nlp = struct('x',obj.x, 'p',obj.p, 'g', obj.g, 'f', obj.f);
            obj.solver = casadi.nlpsol( ...
                ['solver_', obj.name], 'ipopt', nlp, opts);
            
            % save options
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

        function [sol, info]=solve(obj, pars, vals, shift_vals, multistart)
            % SOLVE. Solve the NMPC problem (can be multiple problems) with
            % the given parameter values and initial conditions for the 
            % variables.
            arguments
                obj (1, 1) rlmpc.NMPC
                pars (:, 1) struct
                vals (:, 1) struct
                shift_vals (1, 1) logical = true;
                multistart (1, 1) {mustBePositive,mustBeInteger} = 1
            end
            assert(length(pars) == length(vals), ...
                                    'pars and vals with different sizes')
            
            for i = length(pars)
                % if requested, shifts the initial conditions for the MPC 
                % by M instants to the left, and pads the right with a 
                % simulation. This step is independent of the algorithm and
                % multistart.
                if shift_vals
                    vals(i) = obj.shift_vals(pars(i), vals(i), ...
                                            obj.M, obj.Np, obj.dynamics.f);
                end

                % help with feasibility issues
                vals(i) = obj.impose_feasibility(vals(i));
    
                % order pars and vars according to the order of creation
                vals(i) = orderfields(vals(i), obj.vars);
                pars(i) = orderfields(pars(i), obj.pars);
            end

            % run nlp
            [sol, info] = obj.solve_nlp(pars, vals, multistart);
        end
    end



    methods (Access = protected)
        function [sol, info] = solve_nlp(obj, pars, vals, multistart)
            % pre-compute stuff to avoid obj in the parfor loop
            N = length(pars); % number of multiproblems
            slvr = obj.solver;
            p_ = arrayfun( ...
                @(i) obj.subsevalf(obj.p, obj.pars, pars(i)), 1:N, ...
                'UniformOutput', false);
            lbx_ = obj.lbx;
            ubx_ = obj.ubx;
            lbg_ = obj.lbg;
            ubg_ = obj.ubg;
            varnames = fieldnames(obj.vars)';

            % call NLP solver 
            if multistart == 1 && N == 1 
                % convert to NLP and solve
                x0 = obj.subsevalf(obj.x, obj.vars, vals);
                x0 = max(lbx_, min(ubx_, x0));
                sol = slvr('x0', x0, 'p', p_{1}, 'lbx', lbx_, ...
                          'ubx', ubx_, 'lbg', lbg_, 'ubg', ubg_); 

                % get return status
                status = slvr.stats.return_status;
                success = obj.is_nlp_ok(status);

                % from the unique lam_x, extract lam_lbx and lam_ubx
                lam_lbx_ = -min(0, sol.lam_x);
                lam_ubx_ =  max(0, sol.lam_x);
    
                % build info
                S = [obj.p; obj.x; obj.lam_g; obj.lam_lbx; obj.lam_ubx];
                D = [p_{1}; sol.x; sol.lam_g; lam_lbx_; lam_ubx_];
                get_value = @(o) rlmpc.NMPC.subsevalf(o, S, D);
                info = struct('f', full(sol.f), 'msg', status, ...
                              'success', success, 'get_value', get_value);  
            
                % compute per-variable solution
                sol = struct;
                for n = varnames
                    sol.(n{1}) = get_value(obj.vars.(n{1}));
                end
                return
            end

            % in this case, the NLP is solved in parallel
            sols = cell(1, N * multistart);
            statuses = cell(1, N * multistart);
            parfor i = 1:(N * multistart)
                [np, ns] = ind2sub([N, multistart], i);

                vals_ = rlmpc.NMPC.perturb_vals(vals(np), ns - 1);
                x0 = cellfun(@(n) vals_.(n)(:), varnames, ...
                                            'UniformOutput', false);
                x0 = vertcat(x0{:});
                x0 = max(lbx_, min(ubx_, x0));
                sol_ = slvr('x0', x0, 'p', p_{np}, 'lbx', lbx_, ...
                               'ubx', ubx_, 'lbg', lbg_, 'ubg', ubg_);
                sol_.f = full(sol_.f);
                sols{i} = sol_;
                statuses{i} = slvr.stats.return_status;
            end
            sols = reshape(sols, [N, multistart]);
            statuses = reshape(statuses, [N, multistart]);

            % do these steps for all problems
            successes = obj.is_nlp_ok(statuses);
            for np = 1:N
                % find best among all solutions
                for i = 1:multistart
                    sol_i = sols{np, i};
                    success_i = successes(np, i);
                    if (i == 1) || ...                                  % start with the first one
                       (~success && success_i) || ...                   % pick first that is feasible
                       (success == success_i && sol_i.f < sol_opt.f)    % if both (in)feasible, compare f                                                             
                        sol_opt = sol_i;
                        success = success_i; 
                        i_opt = i;
                    end
                end
                status = statuses{np, i_opt};

                % from the unique lam_x, extract lam_lbx and lam_ubx
                lam_lbx_ = -min(0, sol_opt.lam_x);
                lam_ubx_ =  max(0, sol_opt.lam_x);

                % build info
                S = [obj.p; obj.x; obj.lam_g; obj.lam_lbx; obj.lam_ubx];
                D = [p_{np}; sol_opt.x; sol_opt.lam_g; lam_lbx_; lam_ubx_];
                get_value = @(o) rlmpc.NMPC.subsevalf(o, S, D);
                info(np) = struct('f', full(sol_opt.f), 'msg', status, ...
                              'success', success, 'i_opt', i_opt,...
                              'get_value', get_value);  
            
                % compute per-variable solution
                sol_ = struct;
                for n = varnames
                    sol_.(n{1}) = get_value(obj.vars.(n{1}));
                end
                sol(np) = sol_;
            end
            return
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

        function vals = perturb_vals(vals, std)
            % perturb initial conditions by some magnitude
            vals.w = vals.w + randn(size(vals.w)) * std;
            vals.rho = vals.rho + randn(size(vals.v)) * std * 1.25;
            vals.v = vals.v + randn(size(vals.v)) * std * 1.5;
            vals.r = vals.r + randn(size(vals.r)) * std * 2;
        end

        function vals = impose_feasibility(vals)
            % for some variables, make sure they are nonnegative
            vals.w = max(0, vals.w);
            vals.rho = max(0, vals.rho);
            vals.v = max(0, vals.v);

            % any slack should start at zero
            for n = fieldnames(vals)'
                if startsWith(n{1}, 'slack')
                    vals.(n{1}) = zeros(size(vals.(n{1})));
                end
            end
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

        function ok = is_nlp_ok(status)
            ok = strcmp(status, 'Solve_Succeeded') | ...
                strcmp(status, 'Solved_To_Acceptable_Level');
        end
    end
end
