function F = get_dynamics(n_links, n_origins, n_ramps, n_dist, ...
        T, L, lanes, C, rho_max, tau, delta, eta, kappa, eps)
    % F = GET_DYNAMICS(T, L, lanes, C, rho_max, tau, delta, eta, kappa, eps) 
    %   Creates a casadi.Function object represeting the dynamics equation 
    %   of the 3-link traffic network

    if nargin < 14
        eps = 0; % nonnegative constraint precision
    end

    % states, input and disturbances
    w = casadi.SX.sym('w', n_origins, 1);
    rho = casadi.SX.sym('rho', n_links, 1);
    v = casadi.SX.sym('v', n_links, 1);
    r = casadi.SX.sym('r', n_ramps, 1);
    d = casadi.SX.sym('d', n_dist, 1);

    % parameters
    a = casadi.SX.sym('a', 1, 1);
    v_free = casadi.SX.sym('v_free', 1, 1);
    rho_crit = casadi.SX.sym('rho_crit', 1, 1);

    % ensure nonnegativity - of the input
    w_ = max(eps, w);
    rho_ = max(eps, rho);
    v_ = max(eps, v);
    
    % run system dynamics function
    [q_o, w_o_next, q, rho_next, v_next] = f(w_, rho_, v_, r, d, T, L, ...
        lanes, C, rho_crit, rho_max, a, v_free, tau, delta, eta, kappa, eps);

%     % run system dynamics function
%     [q_o, w_o_next, q, rho_next, v_next] = f(w, rho, v, r, d, T, L, ...
%         lanes, C, rho_crit, rho_max, a, v_free, tau, delta, eta, kappa, eps);

    % ensure nonnegativity - of the output
    q_o = max(eps, q_o);
    w_o_next = max(eps, w_o_next);
    q = max(eps, q);
    rho_next = max(eps, rho_next);
    v_next = max(eps, v_next);

    % create casadi function
    F = casadi.Function('F', {w, rho, v, r, d, a, v_free, rho_crit}, ...
        {q_o, w_o_next, q, rho_next, v_next}, ...
        {'w', 'rho', 'v', 'r', 'd', 'a', 'v_free', 'rho_crit'}, ...
        {'q_o', 'w_o_next', 'q', 'rho_next', 'v_next'});
end


%% local functions
function [q_o, w_o_next, q, rho_next, v_next] = f(w, rho, v, r, d, ...
    T, L, lanes, C, rho_crit, rho_max, a, v_free, tau, delta, eta, kappa, eps)
    %%% ORIGIN
    % compute flow at mainstream origin O1
    q_O1 = min(d(1) + w(1) / T, C(1) * ...
           min(r(1), (rho_max - rho(1)) / (rho_max - rho_crit)));

    % compute flow at onramp origin O2
    q_O2 = min(d(2) + w(2) / T, C(2) * ...
                   min(r(2), (rho_max - rho(3)) / (rho_max - rho_crit)));

    % step queue at origins O1 and O2
    q_o = [q_O1; q_O2];
    w_o_next = w + T * (d(1:2) - q_o);


    %%% BOUNDARIES
    % compute link flows
    q = lanes * rho .* v;

    % compute upstream flow
    q_up = [q_O1; q(1); q(2) + q_O2];

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

    % compute V
    V = Veq(rho, v_free, a, rho_crit, eps);

    % step the speeds of the links
    v_next = (v ...
              + T / tau * (V - v) ...
              + T / L * v .* (v_up - v) ...
              - eta * T / tau / L * (rho_down - rho) ./ (rho + kappa));
    v_next(3) = v_next(3) - delta * T / L / lanes * q_O2 * v(3) / (rho(3) + kappa);    
end


function V = Veq(rho, v_free, a, rho_crit, eps)
    V = v_free * exp((-1 / a) * ((rho / rho_crit) + eps).^a);
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
