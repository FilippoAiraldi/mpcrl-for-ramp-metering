function [q_o, w_o_next, q, rho_next, v_next] = f(w, rho, v, r2, d, config)
    % steps the system by one time step. Returns the next state and other
    % things to save.
    arguments
        w (2, 1) 
        rho (3, 1) 
        v (3, 1) 
        r2 (1, 1) 
        d (2, 1) 
        config
    end

    % unpack config parameters
    T = config.T;
    L = config.L;
    lanes = config.lanes;
    C2 = config.C2;
    rho_crit = config.rho_crit;
    rho_max = config.rho_max;    
    a = config.a;
    v_free = config.v_free;
    tau = config.tau;
    delta = config.delta;
    eta = config.eta;
    kappa = config.kappa;


    %% ORIGIN 
    % compute flow at mainstream origin O1
    V_rho_crit = metanet.Veq(rho_crit, v_free, a, rho_crit);
    v_lim1 = v(1);
    q_cap1 = lanes * V_rho_crit * rho_crit;
    q_speed1 = lanes * v_lim1 * rho_crit * (-a * log(v_lim1 / v_free))^(1 / a);
    q_lim1 = metanet.if_else(v_lim1 < V_rho_crit, q_speed1, q_cap1);
    q_O1 = min(d(1) + w(1) / T, q_lim1);

    % compute flow at onramp origin O2
    q_O2 = min(d(2) + w(2) / T, C2 * ...
                   min(r2, (rho_max - rho(3)) / (rho_max - rho_crit)));

    % step queue at origins O1 and O2
    q_o = [q_O1; q_O2];
    w_o_next = w + T * (d - q_o);


    %% BOUNDARIES
    % compute link flows
    q = lanes * rho .* v;

    % compute upstream flow
    q_up = [q_O1; q(1); q(2) + q_O2];

    % compute upstream speed
    v_up = [v(1); v(1); v(2)];

    % compute downstream density
    rho_down = [rho(2); rho(3); min(rho(3), rho_crit)];
    

    %% LINKS
    % step link densities
    rho_next = rho + (T / (L * lanes)) * (q_up - q);

    % compute V
    V = metanet.Veq(rho, v_free, a, rho_crit);

    % step the speeds of the links
    v_next = (v ...
              + T / tau * (V - v) ...
              + T / L * v .* (v_up - v) ...
              - eta * T / tau / L * (rho_down - rho) ./ (rho + kappa));
    v_next(3) = v_next(3) - delta * T / L / lanes * q_O2 * v(3) / (rho(3) + kappa);    
    

    %% OUTPUTS
    q_o = max(0, q_o);
    w_o_next = max(0, w_o_next);
    q = max(0, q);
    rho_next = max(0, rho_next);
    v_next = max(0, v_next);
end
