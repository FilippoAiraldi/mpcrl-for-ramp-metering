clc, clear all, close all

warning('off')
% load LICQsuccess1.mat
load LICQfail1.mat
warning('on')

% data
ng = numel(g);
dg = full(dg);
dg_kernel = null(dg);

% active constraints
tol = 1e-8;
g_act_idx = find(g > -tol);

% rows of active constraints
dg_act = full(dg(g_act_idx, :));
dg_act_kernel = null(dg_act);

% plotting
tiledlayout('flow')
nexttile
plot(1:ng, g, '*', g_act_idx, g(g_act_idx), 'ro')
legend('constraints', 'active')
labels('ng', 'g(x)')

nexttile
spy(dg_act_kernel)
title('active constraint null space')


