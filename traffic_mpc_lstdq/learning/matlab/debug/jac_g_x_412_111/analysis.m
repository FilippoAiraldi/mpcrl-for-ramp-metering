clc, close all, clear all

% load stuff
load values.mat
f = casadi.Function.load('Jconfunc.cs');
x_sym = casadi.SX.sym('x', size(x));
p_sym = casadi.SX.sym('p', size(p));
y = f(x_sym, p_sym);

% pars
rho_max = 180;
kappa= 40;
a.sym = p_sym(135);
a.val = p(135);
v_free.sym = p_sym(136);
v_free.val = p(136);
rho_crit.sym = p_sym(137);
rho_crit.val = p(137);
d_2_1.sym = p_sym(1);
d_2_1.val = p(1);
% vars
w_2_1.sym = x_sym(2);
w_2_1.val = x(2);
r_1.sym = x_sym(87);
r_1.val = x(87);
rho_2_1.sym = x_sym(92);
rho_2_1.val = x(92);
v_3_1.sym = x_sym(221);
v_3_1.val = x(221);

t4 = d_2_1.val + w_2_1.val / (10 / 3600);
t6 = 2e3 * min(r_1.val, (rho_max - rho_2_1.val) / (rho_max - rho_crit.val));
t7 = rho_2_1.val + kappa;

% term1 = (((v_3_1.val + ( 0.555556 * ((v_free.val * exp(((-1 / a.val) * (rho_2_1.val / rho_crit.val)^a.val)))) - v_3_1.val))) ...
%     +((@1*x_220)*(x_219-x_220)))-((33.3333*(fmax(fmin(x_91,p_136),p_2)-x_91))/(x_91+@2)))-(((@3*fmin(@4,@6))*x_220)/@7)


% y = casadi.substitute(y, p_sym(13), p(13));
% y = casadi.substitute(y, x_sym(9), x(9));
% disp(y)