function dyn = get_dynamics(model, T)
    % GET_DYNAMICS. Creates a structure containing the real and nominal 
    % dynamics (casadi.Function and its variables) representing the 
    % underlying dynamics of the 3-link traffic network
    arguments
        model (1, 1) struct
        T (1, 1) double {mustBePositive}
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
                                        rho_crit, a, v_free, T, model);

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



%% local functions
function [q_o, w_o_next, q, rho_next, v_next] = f( ...
                            w, rho, v, r, d, rho_crit, a, v_free, T, model)
    % Computes the actual dynamical equations. It can work both with
    % symbolical variables and numerical variables
    C = model.C;
    lanes = model.lanes;
    rho_max = model.rho_max;
    L = model.L;
    tau = model.tau;
    eta = model.eta;
    kappa = model.kappa;
    delta = model.delta;

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
