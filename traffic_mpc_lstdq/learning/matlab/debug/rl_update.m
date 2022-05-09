clc, clearvars, close all,

warning('off')
load rl_update.mat
warning('on')

% vanilla
delta0 = lr * sample.b / sample.n;

% 1st order Q learning update
f = -lr * sample.b / sample.n; 
H = eye(length(f));
[~, delta1] = rlmpc.rl_constr_update(rl_pars, rl_pars_bounds, H, f);

% 2nd order Q learning update 1
f = lr * 100 * sample.b;
H = (sample.A + sample.A') / 2 + 1e6 * eye(size(sample.A));
[~, delta2] = rlmpc.rl_constr_update(rl_pars, rl_pars_bounds, H, f);

% 2nd order Q learning update 2
f = lr * 100 * ((sample.A + 1e-6 * eye(size(sample.A))) \ sample.b);
% f = lr * (sample.A \ sample.b);
H = eye(length(f));
[~, delta3] = rlmpc.rl_constr_update(rl_pars, rl_pars_bounds, H, f);

k = 1:length(delta0);
plot(k, abs(delta0), k, abs(delta1), k, abs(delta2), k, abs(delta3)), 
legend('vanilla', '1st order', '2nd order H+f', '2nd order H\\f')
