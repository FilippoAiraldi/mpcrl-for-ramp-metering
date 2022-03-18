function F = get_dynamics(T, L, lanes, C2, rho_max, tau, delta, eta, kappa)
    % F = GET_DYNAMICS(T, L, lanes, C2, rho_max, tau, delta, eta, kappa) 
    %   Creates a casadi.Function object represeting the dynamics equation 
    %   of the 3-link traffic network

    % states, input and disturbances
    w = casadi.SX.sym('w', 2, 1);
    rho = casadi.SX.sym('rho', 3, 1);
    v = casadi.SX.sym('v', 3, 1);
    r2 = casadi.SX.sym('r', 1, 1);
    d = casadi.SX.sym('d', 2, 1);

    % parameters
    a = casadi.SX.sym('a', 1, 1);
    v_free = casadi.SX.sym('v_free', 1, 1);
    rho_crit = casadi.SX.sym('rho_crit', 1, 1);

    % run system dynamics function
    [q_o, w_o_next, q, rho_next, v_next] = f(w, rho, v, r2, d, T, L, ...
        lanes, C2, rho_crit, rho_max, a, v_free, tau, delta, eta, kappa);

    % create casadi function
    F = casadi.Function('F', {w, rho, v, r2, d, a, v_free, rho_crit}, ...
        {q_o, w_o_next, q, rho_next, v_next}, ...
        {'w', 'rho', 'v', 'r', 'd', 'a', 'v_free', 'rho_crit'}, ...
        {'q_o', 'w_o_next', 'q', 'rho_next', 'v_next'});
end


%% local functions
function [q_o, w_o_next, q, rho_next, v_next] = f(w, rho, v, r2, d, ...
    T, L, lanes, C2, rho_crit, rho_max, a, v_free, tau, delta, eta, kappa)
    %%% ORIGIN
    % compute flow at mainstream origin O1
    V_rho_crit = Veq(rho_crit, v_free, a, rho_crit);
    v_lim1 = v(1);
    q_cap1 = lanes * V_rho_crit * rho_crit;
    q_speed1 = lanes * v_lim1 * rho_crit * (-a * log(v_lim1 / v_free))^(1 / a);
    q_lim1 = if_else(v_lim1 < V_rho_crit, q_speed1, q_cap1);
    q_O1 = min(d(1) + w(1) / T, q_lim1);

    % compute flow at onramp origin O2
    q_O2 = min(d(2) + w(2) / T, C2 * ...
                   min(r2, (rho_max - rho(3)) / (rho_max - rho_crit)));

    % step queue at origins O1 and O2
    q_o = [q_O1; q_O2];
    w_o_next = w + T * (d - q_o);


    %%% BOUNDARIES
    % compute link flows
    q = lanes * rho .* v;

    % compute upstream flow
    q_up = [q_O1; q(1); q(2) + q_O2];

    % compute upstream speed
    v_up = [v(1); v(1); v(2)];

    % compute downstream density
    rho_down = [rho(2); rho(3); min(rho(3), rho_crit)];


    %%% LINK
    % step link densities
    rho_next = rho + (T / (L * lanes)) * (q_up - q);

    % compute V
    V = Veq(rho, v_free, a, rho_crit);

    % step the speeds of the links
    v_next = (v ...
              + T / tau * (V - v) ...
              + T / L * v .* (v_up - v) ...
              - eta * T / tau / L * (rho_down - rho) ./ (rho + kappa));
    v_next(3) = v_next(3) - delta * T / L / lanes * q_O2 * v(3) / (rho(3) + kappa);    


    %%% OUT
    q_o = max(0, q_o);
    w_o_next = max(0, w_o_next);
    q = max(0, q);
    rho_next = max(0, rho_next);
    v_next = max(0, v_next);
end

function V = Veq(rho, v_free, a, rho_crit)
    % VEQ Evaluates the METANET speed equation at the given density rho.
    
    V = v_free * exp((-1 / a) * (rho / rho_crit).^a);
end

function varargout = if_else(varargin)
    %IF_ELSE Branching on MX nodes Ternary operator, "cond ? if_true : if_false".
    %
    %  DM = IF_ELSE(DM cond, DM if_true, DM if_false, bool short_circuit)
    %  SX = IF_ELSE(SX cond, SX if_true, SX if_false, bool short_circuit)
    %  MX = IF_ELSE(MX cond, MX if_true, MX if_false, bool short_circuit)
    %
    %  Had to manually pick this function from the original CasADi Matlab 
    % installation folder since it is missing from its namespace.
    
    [varargout{1:nargout}] = casadiMEX(235, varargin{:});
end

