function dyn = get_dynamics( ...
                n_links, n_origins, n_ramps, n_dist, ...
                T, L, lanes, C, rho_max, tau, delta, eta, kappa, ...
                max_in_and_out, eps, ...
                origin_as_ramp, control_origin, ...
                simplified_rho_down, flow_as_control_action, Veq_approx)
    % GET_DYNAMICS. Creates a structure containing the real and nominal 
    % dynamics (casadi.Function and its variables) representing the 
    % underlying dynamics of the 3-link traffic network
    arguments
        n_links, n_origins, n_ramps, n_dist ...
            (1, 1) double {mustBePositive,mustBeInteger}
        T, L (1, 1) double {mustBePositive}
        lanes (1, 1) double {mustBePositive,mustBeInteger}
        C (:, 1) double {mustBeNonnegative}
        rho_max, tau, delta, eta, kappa (1, 1) double {mustBeNonnegative}
        max_in_and_out (1, 2) logical = [false, false]
        eps (1, 1) double {mustBeNonnegative} = 0
        origin_as_ramp (1, 1) logical = false
        control_origin (1, 1) logical = false
        simplified_rho_down (1, 1) logical = false
        flow_as_control_action (1, 1) logical = false
        Veq_approx (1, 1) casadi.Function = casadi.Function
    end
    assert(numel(C) == n_origins, 'origins capacities and number mismatch')

    dyn = struct;
    for name = ["real", "nominal"]
        use_Veq_approx = name == "nominal" && ~Veq_approx.is_null;
        if name == "real"
            eps_ = 0;
        else
            eps_ = eps;
        end

        % create states, input, disturbances and other parameters
        w = casadi.SX.sym('w', n_origins, 1);
        rho = casadi.SX.sym('rho', n_links, 1);
        v = casadi.SX.sym('v', n_links, 1);
        r = casadi.SX.sym('r', n_ramps, 1);
        d = casadi.SX.sym('d', n_dist, 1); 
        a = casadi.SX.sym('a', 1, 1);
        v_free = casadi.SX.sym('v_free', 1, 1);
        rho_crit = casadi.SX.sym('rho_crit', 1, 1);
        if ~use_Veq_approx
            Veq = [];
        else
            Veq.f = Veq_approx;
            Veq.pars = casadi.SX.sym( ...
                        Veq_approx.name_in(1), size(Veq_approx.mx_in(1)));
        end
    
        % ensure nonnegativity of the input
        % run system dynamics function (max for nonnegativity of inputs 
        % and outputs)
        states = {w, rho, v};
        if name == "real" || max_in_and_out(1)
           for i = length(states) 
               states{i} = max(eps_, states{i});
           end
        end
        [q_o, w_o_next, q, rho_next, v_next] = f( ...
            states{:}, r, d, ...
            T, L, lanes, C, rho_max, ...
            rho_crit, a, v_free, ...
            tau, delta, eta, kappa, eps_, ...
            origin_as_ramp, control_origin, simplified_rho_down, ...
            flow_as_control_action, Veq);
        if name == "real" || max_in_and_out(2) % real dynamics should not ouput negatives
            q_o = max(eps_, q_o);
            w_o_next = max(eps_, w_o_next);
            q = max(eps_, q);
            rho_next = max(eps_, rho_next);
            v_next = max(eps_, v_next);
        end
    
        % create dynamics function args and outputs
        args = struct('w', w, 'rho', rho, 'v', v, 'r', r, 'd', d, ...
            'rho_crit', rho_crit);
        if ~use_Veq_approx
            args.a = a;
            args.v_free = v_free;
        else
            args.pars_Veq_approx = Veq.pars;
        end
        out = struct('q_o', q_o, 'w_o_next', w_o_next, 'q', q, ...
            'rho_next', rho_next, 'v_next', v_next);
    
        % create casadi function and division between vars and pars
        dyn.(name) = struct;
        dyn.(name).f = casadi.Function('F', ...
            struct2cell(args), struct2cell(out), ...
            fieldnames(args), fieldnames(out));
        dyn.(name).states = struct('w', w, 'rho', rho, 'v', v); % states
        dyn.(name).input = struct('r', r);                      % controls
        dyn.(name).dist = struct('d', d);                       % disturbances
        dyn.(name).pars = struct;                               % parameters
        remaining_args = setdiff(fieldnames(args), ...
                                [fieldnames(dyn.(name).states); ...
                                fieldnames(dyn.(name).input); ...
                                fieldnames(dyn.(name).dist)], 'stable');
        for arg = remaining_args'
            dyn.(name).pars.(arg{1}) = args.(arg{1});
        end
    end

    % dynamics
    % F:(w[2],rho[3],v[3],r,d[3],rho_crit,a,v_free)->(q_o[2],w_o_next[2],q[3],rho_next[3],v_next[3])
    % dynamics with Veq approx
    % F:(w[2],rho[3],v[3],r,d[3],rho_crit,pars_Veq_approx)->(q_o[2],w_o_next[2],q[3],rho_next[3],v_next[3])
end



%% local functions
function [q_o, w_o_next, q, rho_next, v_next] = f( ...
            w, rho, v, r, d, ...
            T, L, lanes, C, rho_max, ...
            rho_crit, a, v_free, ...
            tau, delta, eta, kappa, ...
            eps, origin_as_ramp, control_origin, ...
            simplified_rho_down, ...
            flow_as_control_action, Veq_approx)
    % Computes the actual dynamical equations. It can work both with
    % symbolical variables and numerical variables

    % which link the on-ramp is attached to
    ramped_link = 3;


    %%% ORIGIN
    if origin_as_ramp
        % if the origin is not controlled, then its ramp rate is full
        if ~control_origin
            r = [1; r];
            if flow_as_control_action
                r(1) = min(d(1) + w(1) / T, C(1) * ...
                        min(1, (rho_max - rho(1)) / (rho_max - rho_crit)));
            end
        end

        if flow_as_control_action
            % the flow of the ramp is the control itself
            q_O1 = r(1);
            q_O2 = r(2);
        else
            % compute flow at mainstream origin O1
            q_O1 = r(1) * min(d(1) + w(1) / T, C(1) * ...
                     min(1, (rho_max - rho(1)) / (rho_max - rho_crit)));

            % compute flow at onramp origin O2 (NOTE: this formulation uses 2 mins,
            % whereas the min of 3-element vector uses 3 mins)
            q_O2 = r(2) * min(d(2) + w(2) / T, C(2) * ...
                    min(1, ...
                     (rho_max - rho(ramped_link)) / (rho_max - rho_crit)));
        end
    
        % step queue at origins O1 and O2
        q_o = [q_O1; q_O2];
        w_o_next = w + T * (d(1:2) - q_o);        
    else
        % origin flow is just the demand
        q_O1 = d(1);
    
        if flow_as_control_action
            % the flow of the ramp is the control itself
            q_O2 = r;
        else
            % compute flow at onramp origin O2 (NOTE: this formulation uses 2 mins,
            % whereas the min of 3-element vector uses 3 mins)
            q_O2 = min(d(2) + w(1) / T, C * ...
                    min(r, ...
                     (rho_max - rho(ramped_link)) / (rho_max - rho_crit)));
        end
    
        % step queue at origins O1 and O2
        q_o = q_O2;
        w_o_next = w + T * (d(2) - q_o);                
    end


    %%% BOUNDARIES
    % compute link flows
    q = lanes * rho .* v;

    % compute upstream flow
    q_up = [q_O1; q(1); q(2)];
    q_up(ramped_link) = q_up(ramped_link) + q_O2;

    % compute upstream speed
    v_up = [v(1); v(1); v(2)];

    % compute downstream density
    if length(d) > 2
        if simplified_rho_down
            rho_down3 = d(3);
        else
            rho_down3 = max(min(rho(3), rho_crit), d(3));
        end
    else
        if simplified_rho_down
            rho_down3 = rho(3);
        else
            rho_down3 = min(rho(3), rho_crit);
        end
        
    end
    rho_down = [rho(2); rho(3); rho_down3];


    %%% LINK
    % step link densities
    rho_next = rho + (T / (L * lanes)) * (q_up - q);

    % compute equilibrium V - decide if Veq should be approximated
    if isempty(Veq_approx)
        Veq_ = metanet.Veq(rho, v_free, a, rho_crit, eps);
    else
        Veq_ = arrayfun(@(x) Veq_approx.f(x, Veq_approx.pars), rho, ...
                                                'UniformOutput', false);
        Veq_ = vertcat(Veq_{:});
    end

    % step the speeds of the links
    v_next = (v ...
              + T / tau * (Veq_ - v) ...
              + T / L * v .* (v_up - v) ...
              - eta * T / tau / L * (rho_down - rho) ./ (rho + kappa));
    v_next(ramped_link) = v_next(ramped_link) ...
        - delta * T / L / lanes * ...
            q_O2 * v(ramped_link) / (rho(ramped_link) + kappa);    
end
