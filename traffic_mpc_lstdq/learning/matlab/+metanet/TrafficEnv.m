classdef TrafficEnv < handle
    % TRAFFICENV. OpenAI-like gym environment for METANET traffic control
    %  applications.

    properties (GetAccess = public, SetAccess = protected)
        episodes (1, 1) double {mustBePositive, mustBeInteger} = 50
        sim (1, 1) struct
        model (1, 1) struct
        dynamics (1, 1) struct
        %
        state (1, 1) struct
        demand (:, :) double {mustBeNonnegative}
        %
        r_prev (:, 1) double 
        %
        k_tot (1, 1) double {mustBePositive,mustBeInteger} = 1 % internal total time counter
    end
    % history.origins.queue{iteration}{episode}(time step in the episode)

    properties (Dependent) 
        x (:, 1) double % state but as a vector
        d (:, 1) double % current demand
        %
        ep (1, 1) double {mustBePositive,mustBeInteger} % internal counter of episode
        k (1, 1) double {mustBePositive,mustBeInteger} % internal time counter per episode
    end


    
    methods (Access = public)
        function obj = TrafficEnv(episodes, sim, model)
            % TRAFFICENV. Constructs an empty instance of this gym.
            arguments
                episodes (1, 1) double {mustBePositive, mustBeInteger}
                sim (1, 1) struct
                model (1, 1) struct
            end
            obj.episodes = episodes;
            obj.sim = sim;
            obj.model = model;
            obj.dynamics = get_dynamics(sim, model);
        end

        function reset(obj, r)
            % RESET. Resets the environment to default, random conditions.
            arguments
                obj (1, 1) metanet.TrafficEnv
                r (:, 1) double = obj.model.C(2)
            end
            mdl = obj.model;

            % create random demands for all episodes + 1 to avoid 
            % out-of-bound access
            obj.demand = get_demand_profiles( ...
                                    obj.sim.t, obj.episodes + 1, 'fixed');
            assert(size(obj.demand, 1) == mdl.n_dist)

            % compute initial state (at steady-state)
            [w, rho, v] = steady_state(obj.dynamics.f, ...
                                       zeros(mdl.n_origins, 1), ...
                                       10 * ones(mdl.n_links, 1), ...
                                       100 * ones(mdl.n_links, 1), ...
                                       r, obj.demand(:, 1), ...
                                       mdl.rho_crit,mdl.a,mdl.v_free);
            obj.state = struct('w', w, 'rho', rho, 'v', v);

            % reset time counter
            obj.k_tot = 1;
            obj.r_prev = r;
        end

        function [state, cost, done, info] = step(obj, r)
            % STEP. Steps the discrete dynamics by one time step, given a 
            % control ramp flow.
            arguments
                obj (1, 1) metanet.TrafficEnv
                r (:, 1) double % numerical or symbolical
            end
            assert(size(r, 1) == obj.model.n_ramps)
            
            % step the dynamics (according to the true parameters)
            [q_o, w_next, q, rho_next, v_next] = obj.dynamics.f(...
                        obj.state.w, obj.state.rho, obj.state.v, ... 
                        r, obj.demand(:, obj.k_tot), ...
                        obj.model.rho_crit, obj.model.a, obj.model.v_free);

            % save as state
            state = struct('w', full(w_next), 'rho', full(rho_next), ...
                'v', full(v_next));
            info = struct('q', full(q), 'q_o', full(q_o));

            % ...TODO: COMPUTE COST...
            cost = nan;

            % is the episode over?
            done = obj.k_tot == obj.sim.K * obj.episodes; % or the previous iter?
            
            % save quantities for next iteration
            obj.k_tot = obj.k_tot + 1;
            obj.r_prev = r;
            obj.state = state;
        end
    end

    methods
        function x = get.x(obj)
            x = cell2mat(struct2cell(obj.state));
        end

        function d = get.d(obj)
            d = obj.demand(:, obj.k_tot);
        end

        function k = get.k(obj)
            k = mod(obj.k_tot - 1, obj.sim.K) + 1;
        end

        function ep = get.ep(obj)
            ep = ceil(obj.k_tot / obj.sim.K);
        end
    end
end



%% local functions for demand profile generation
function D = get_demand_profiles(time, episodes, type)
    % GET_DEMAND_PROFILES. Returns the demand profiles for the 
    % network, either fixed or random.
    arguments
        time (1, :) double
        episodes (1, 1) {mustBeInteger, mustBePositive}
        type {mustBeMember(type, {'fixed', 'random'})} = 'random'
    end

    if strcmp(type, 'fixed')
        d1 = create_profile(time, [0, .35, 1, 1.35], [1e3, 3e3, 3e3, 1e3]);
        d2 = create_profile(time, [.15,.35,.6,.8], [500, 1500, 1500, 500]);
        d_cong = create_profile(time, [0.5, .7, 1, 1.2], [20, 60, 60, 20]);
        D = repmat([d1; d2; d_cong], 1, episodes);
    else
        d1_h = rand_between([1, episodes], 2750, 3500);
        d1_c = rand_between([1, episodes], 0.5, 0.9);
        d1_w1  = rand_between([1, episodes], 0.2, 0.4) / 2;
        d1_w2  = rand_between(size(d1_w1), d1_w1 * 1.5, d1_w1 * 3);
        d2_h = rand_between([1, episodes], 1250, 2000);
        d2_c = rand_between([1, episodes], 0.3, 0.7);
        d2_w1  = rand_between([1, episodes], 0.1, 0.4) / 2;
        d2_w2  = rand_between(size(d2_w1), d2_w1 * 1.5, d2_w1 * 3);
        d3_h = rand_between([1, episodes], 45, 80);
        d3_c = rand_between([1, episodes], 0.7, 1);
        d3_w1  = rand_between([1, episodes], 0.1, 0.5) / 2;
        d3_w2  = rand_between(size(d3_w1), d3_w1 * 1.5, d3_w1 * 3);
        
        D = cell(3, episodes);
        for i = 1:episodes
            D{1, i} = create_profile(time, ...
                [d1_c(i) - d1_w2(i), d1_c(i) - d1_w1(i), ...
                               d1_c(i) + d1_w1(i), d1_c(i) + d1_w2(i)], ...
                [1000, d1_h(i), d1_h(i), 1000]);
            D{2, i} = create_profile(time, ...
                [d2_c(i) - d2_w2(i), d2_c(i) - d2_w1(i), ...
                               d2_c(i) + d2_w1(i), d2_c(i) + d2_w2(i)], ...
                [500, d2_h(i), d2_h(i), 500]);
            D{3, i} = create_profile(time, ...
                [d3_c(i) - d3_w2(i), d3_c(i) - d3_w1(i), ...
                               d3_c(i) + d3_w1(i), d3_c(i) + d3_w2(i)], ...
                [20, d3_h(i), d3_h(i), 20]);
        end
        D = cell2mat(D);
    end
    
    % add noise and then filter to make it more realistic
    [filter_num, filter_den] = butter(3, 0.1);
    D = filtfilt(filter_num, filter_den, ...
                                (D + randn(size(D)) .* [95; 95; 1.7])')';
end

function profile = create_profile(t, x, y)
    % CREATE_PROFILE. Creates a profile passing through points (x, y) along 
    % time t.
    arguments
        t (1, :) double
        x (1, :) double
        y (1, :) double
    end
    assert(isequal(size(x), size(y)), 'x and y do not share the same size')

    x(1) = max(x(1), t(1));
    x(end) = min(x(end), t(end));
    assert(issorted(x), 'x is ill-specified')
    % assert(x(1) >= t(1) && x(end) <= t(end), 'x must contained within t')

    [~, x] = min(abs(t - x'), [], 2);
    if x(1) ~= 1
        x = [1, x', length(t)];
        y = [y(1), y, y(end)];
    else
        x = [x', length(t)];
        y = [y, y(end)];        
    end

    profile = zeros(size(t));
    for i = 1:length(x) - 1
        m = (y(i + 1) - y(i)) / (t(x(i + 1)) - t(x(i)));
        q = y(i) - m * t(x(i));
        profile(x(i):x(i + 1)) = m * t(x(i):x(i + 1)) + q;
    end
    profile(end) = profile(end - 1);
end

function r = rand_between(size, a, b)
    r = rand(size) .* (b - a) + a;
end



%% loacl function for dynamics
function dyn = get_dynamics(sim, model)
    % GET_DYNAMICS. Creates a structure containing the real and nominal 
    % dynamics (casadi.Function and its variables) representing the 
    % underlying dynamics of the 3-link traffic network
    arguments
        sim (1, 1) struct
        model (1, 1) struct
    end
   
    % create states, input, disturbances and other parameters
    w = casadi.SX.sym('w', model.n_origins, 1);
    rho = casadi.SX.sym('rho', model.n_links, 1);
    v = casadi.SX.sym('v', model.n_links, 1);
    r = casadi.SX.sym('r', model.n_ramps, 1);
    d = casadi.SX.sym('d', model.n_dist, 1); 
    a = casadi.SX.sym('a', 1, 1);
    v_free = casadi.SX.sym('v_free', 1, 1);
    rho_crit = casadi.SX.sym('rho_crit', 1, 1);

    % run system dynamics function
    [q_o, w_o_next, q, rho_next, v_next] = f(w, rho, v, r, d, ...
                                        rho_crit, a, v_free, model, sim);

    % make sure that output is not negative
    q_o = max(0, q_o);
    w_o_next = max(0, w_o_next);
    q = max(0, q);
    rho_next = max(0, rho_next);
    v_next = max(0, v_next);

    % create dynamics function args and outputs
    args = struct('w', w, 'rho', rho, 'v', v, 'r', r, 'd', d, ...
        'rho_crit', rho_crit, 'a', a, 'v_free', v_free);
    out = struct('q_o', q_o, 'w_o_next', w_o_next, 'q', q, ...
        'rho_next', rho_next, 'v_next', v_next);

    % create casadi function and division between vars and pars
    dyn = struct;
    dyn.f = casadi.Function('F', struct2cell(args), struct2cell(out), ...
                                 fieldnames(args), fieldnames(out));
    dyn.states = struct('w', w, 'rho', rho, 'v', v);                    % states
    dyn.input = struct('r', r);                                         % controls
    dyn.dist = struct('d', d);                                          % disturbances
    dyn.pars = struct('rho_crit', rho_crit, 'a', a, 'v_free', v_free);  % parameters

    % dynamics
    % F:(w[2],rho[3],v[3],r,d[3],rho_crit,a,v_free)->(q_o[2],w_o_next[2],q[3],rho_next[3],v_next[3])
end

function [q_o, w_o_next, q, rho_next, v_next] = f( ...
                        w, rho, v, r, d, rho_crit, a, v_free, model, sim)
    % F. Computes the actual dynamical equations. It can work both with
    % symbolical variables and numerical variables
    C = model.C;
    lanes = model.lanes;
    rho_max = model.rho_max;
    L = model.L;
    tau = model.tau;
    eta = model.eta;
    kappa = model.kappa;
    delta = model.delta;
    T = sim.T;

    % which link the on-ramp is attached to
    ramped_link = 3;


    %%% ORIGIN
    % the flow of the ramp is the control itself. Since the origin is 
    % not controlled, its ramp rate is full
    q_O1 = min(d(1) + w(1) / T, C(1) * ...
                min(1, (rho_max - rho(1)) / (rho_max - rho_crit)));
    q_O2 = r;

    % step queue at origins O1 and O2
    q_o = [q_O1; q_O2];
    w_o_next = w + T * (d(1:2) - q_o);        


    %%% BOUNDARIES
    % compute link flows
    q = lanes * rho .* v;

    % compute upstream flow
    q_up = [q_O1; q(1); q(2)];
    q_up(ramped_link) = q_up(ramped_link) + q_O2;

    % compute upstream speed
    v_up = [v(1); v(1); v(2)];

    % compute downstream density
    rho_down = [rho(2); rho(3); max(min(rho(3), rho_crit), d(3))];


    %%% LINK
    % step link densities
    rho_next = rho + (T / (L * lanes)) * (q_up - q);

    % step the speeds of the links
    Veq_ = metanet.Veq(rho, v_free, a, rho_crit);
    v_next = (v ...
              + T / tau * (Veq_ - v) ...
              + T / L * v .* (v_up - v) ...
              - eta * T / tau / L * (rho_down - rho) ./ (rho + kappa));
    v_next(ramped_link) = v_next(ramped_link) ...
        - delta * T / L / lanes * ...
            q_O2 * v(ramped_link) / (rho(ramped_link) + kappa);    
end

function [w_ss, rho_ss, v_ss, err, k] = steady_state( ...
                                                F, w0, rho0, v0, r, d, ...
                                                rho_crit, a, v_free, ...
                                                tol, maxiter)
    % STEADY_STATE. Runs the given dynamic function until convergence to a
    % steady-state is detected.
    arguments
        F (1, 1) casadi.Function
        w0, rho0, v0, r, d (:, 1) double
        rho_crit, a, v_free (1, 1) double {mustBeNonnegative}
        tol (1, 1) double {mustBeNonnegative} = 1e-4
        maxiter (1, 1) double {mustBeNonnegative,mustBeInteger} = 1e3
    end

    k = 0;
    err = inf;
    while k < maxiter
        % step the system
        [~, w_ss, ~, rho_ss, v_ss] = F(w0, rho0, v0, r, d, ...
                                                    rho_crit, a, v_free);
        w_ss = full(w_ss);
        rho_ss = full(rho_ss);
        v_ss = full(v_ss);

        % compute convergence error
        err = norm(w_ss - w0) + norm(rho_ss - rho0) + norm(v_ss - v0);
        w0 = w_ss;
        rho0 = rho_ss;
        v0 = v_ss;
        k = k + 1;
        if err < tol
            return
        end
    end
    warning('steady-state not reached; stopped at max iterations')
end
