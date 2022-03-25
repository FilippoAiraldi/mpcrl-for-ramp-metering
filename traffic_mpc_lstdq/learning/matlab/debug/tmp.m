clc, close all, clear all

% load stuff
load values.mat
f = casadi.Function.load('Jconfunc');
x_sym = casadi.SX.sym('x', 390, 1);
p_sym = casadi.SX.sym('p', 151, 1);
y = f(x_sym, p_sym);

% pars
a.sym = p_sym(135);
a.val = p(135);
v_free.sym = p_sym(136);
v_free.val = p(136);
rho_crit.sym = p_sym(137);
rho_crit.val = p(137);
d_1_5.sym = p_sym(13);
d_1_5.val = p(13);
% vars
w_1_5.sym = x_sym(9);
w_1_5.val = x(9);
v_1_5.sym = x_sym(231);
v_1_5.val = x(231);


% y = casadi.substitute(y, p_sym(13), p(13));
% y = casadi.substitute(y, x_sym(9), x(9));
disp(y)