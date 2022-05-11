function dyn = get_dynamics(n_links, n_origins, n_ramps, n_dist, ...
        T, L, lanes, C, rho_max, tau, delta, eta, kappa, ...
        max_in_and_out, eps, Veq_approx)
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
            tau, delta, eta, kappa, eps_, Veq);
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
            tau, delta, eta, kappa, eps, Veq_approx)
    % Computes the actual dynamical equations. It can work both with
    % symbolical variables and numerical variables

    % which link the on-ramp is attached to
    link_with_ramp = 3;
   
    % if only one rate is given, then apply it to the second origin
    if numel(r) == 1
        r = [1; r];
    end

    %%% ORIGIN
    % compute flow at mainstream origin O1
    q_O1 = min(d(1) + w(1) / T, C(1) * ...
           min(r(1), (rho_max - rho(1)) / (rho_max - rho_crit)));

    % compute flow at onramp origin O2
    q_O2 = min(d(2) + w(2) / T, C(2) * ...
        min(r(2), (rho_max - rho(link_with_ramp)) / (rho_max - rho_crit)));

    % step queue at origins O1 and O2
    q_o = [q_O1; q_O2];
    w_o_next = w + T * (d(1:2) - q_o);


    %%% BOUNDARIES
    % compute link flows
    q = lanes * rho .* v;

    % compute upstream flow
    q_up = [q_O1; q(1); q(2)];
    q_up(link_with_ramp) = q_up(link_with_ramp) + q_O2;

    % compute upstream speed
    v_up = [v(1); v(1); v(2)];

    % compute downstream density
    if length(d) > 2
        rho_down = [rho(2); rho(3); max(min(rho(3), rho_crit), d(3))];
    else
        rho_down = [rho(2); rho(3); min(rho(3), rho_crit)];
    end


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
    v_next(link_with_ramp) = v_next(link_with_ramp) ...
        - delta * T / L / lanes * ...
            q_O2 * v(link_with_ramp) / (rho(link_with_ramp) + kappa);    
end


% function [q_o, w_o_next, q, rho_next, v_next] = f(w, rho, v, r2, d, ...
%     T, L, lanes, C, rho_crit, rho_max, a, v_free, tau, delta, eta, kappa, ...
%     eps)
%     %%% ORIGIN
%     % compute flow at mainstream origin O1
%     V_rho_crit = Veq(rho_crit, v_free, a, rho_crit);
%     v_lim1 = v(1);
%     q_cap1 = lanes * V_rho_crit * rho_crit;
%     q_speed1 = lanes * v_lim1 * rho_crit * (-a * log(v_lim1 / v_free + eps))^(1 / a);
%     q_lim1 = if_else(v_lim1 < V_rho_crit, q_speed1, q_cap1);
%     q_O1 = min(d(1) + w(1) / T, q_lim1);
% 
%     % compute flow at onramp origin O2
%     q_O2 = min(d(2) + w(2) / T, C(2) * ...
%                    min(r2, (rho_max - rho(3)) / (rho_max - rho_crit)));
% 
%     % step queue at origins O1 and O2
%     q_o = [q_O1; q_O2];
%     w_o_next = w + T * (d(1:2) - q_o);
% 
% 
%     %%% BOUNDARIES
%     % compute link flows
%     q = lanes * rho .* v;
% 
%     % compute upstream flow
%     q_up = [q_O1; q(1); q(2) + q_O2];
% 
%     % compute upstream speed
%     v_up = [v(1); v(1); v(2)];
% 
%     % compute downstream density
%     if length(d) > 2
%         rho_down = [rho(2); rho(3); max(min(rho(3), rho_crit), d(3))];
%     else
%         rho_down = [rho(2); rho(3); min(rho(3), rho_crit)];
%     end
% 
% 
%     %%% LINK
%     % step link densities
%     rho_next = rho + (T / (L * lanes)) * (q_up - q);
% 
%     % compute V
%     V = Veq(rho, v_free, a, rho_crit);
% 
%     % step the speeds of the links
%     v_next = (v ...
%               + T / tau * (V - v) ...
%               + T / L * v .* (v_up - v) ...
%               - eta * T / tau / L * (rho_down - rho) ./ (rho + kappa));
%     v_next(3) = v_next(3) - delta * T / L / lanes * q_O2 * v(3) / (rho(3) + kappa);    
% end


% function varargout = if_else(varargin)
%     [varargout{1:nargout}] = casadiMEX(235, varargin{:});
% end
