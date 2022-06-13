################# 1st meeting #################
LEARNING
	1. create scenario where there's an evident discrepancy in performance between MPC with perfect knowledge and MPC with modified parameters 
		- modified a 
		- modified learnable rho crit
		- modified learnable v_free

	2. create a learning algorithm 
		- it learns
			- parameters rho_crit and v_free
			- cost function: quadratic in state and control (with rho_crit tracking) 
				-> look at S. Gros' students' papers
		- RL
			- Q learning with experience replay
			- learning rate
			- discount
			- introduce random disturbance in cost to induce exploration?
			- introduce cost for constraint violation, i.e., w^T * max(0, g(x))

	(might need to log intermediate results since simulations will be long)
    
	
	
################# 2nd meeting #################
## QUESTIONS 
0. RL-MPC
	- Simulation scenarion in general: updates RL at the end of each cycle of simulation, like a repetitive task
	- Q-learning: difference between MPC cost function and RL cost function, are they the same or not? Because I have seen that in Gros papers they use the same stage cost
		- if they are the same, then what are the terms? 
			- stage cost
				TTS + input variability + slack penalty (these are from standard MPC) + quadratic term (with rho_crit tracking) in control and states + affine term (in control and states?)
			- other terms
				- initial term (just a simple bias?)
				- final term? (quadratic?)
		- with experience replay (or LSTD), what to save:
			td error
		
		L = TTS + input variability + slack penalty + quadratic term (with rho_crit tracking) in control and states + affine term (in control and states?)
			My issue with this definition of L is that TTS looks into the future (and ok, we can disassemble its sum over k) 
			but the input variability works on another time line, as well as the slack penalty (maybe these 2 should not be in the stage cost but in the final)
		init = bias?
		final = quadratic in which terms?
	
	- first-order or second-order Q-learning? (error in equation 19b, Esfahani)
	- differentiability of MPC holds only when the KKT conditions are met, i.e., solver reached optimality. What to do if the solver does not? Simply don't pay attention to it?



################# 3rd meeting #################
- Q-learning: + or -?
- higher number of iterations
- check 1st KKT condition is satisfied when computing the derivatives of Q: in this way, we know the lagrangian is ok (V)
- start with lower parameters, not higher -> chance of constraint violation should increase
- before running sim, just run the sim with true parameters so that we know how much the TTS is for perfect conditions



################# 4th meeting #################
- replace the equilibrium speed equation with a linear function, since it is quite nonlinear
	- (Figure 3.7 of Hegyi thesis) replace the descending exp with a line. Put it in a max(0, .), because it must not become negative.
	- add an offset c * rho to this equation, with c small and learnable, so that for big values of rho we still have some contribution to the speed.
	- make this term takes effect only for high rhos.. how?
	- TRY REMOVE SOME MAX ON INPUT AND OUTPUTS AND SET EPS=0 TO SEE IF REMOVING THE NONLINEARITY HELPED
- make vector or weights for slacks, instead of a single shared learnable weight (done)
- make the weight of input rate variability cost (the one at 0.4) learnable (done)
- modify the update rule: use Hessian (see Esfahani)



################# 5th meeting #################
- shorter horizon
- LICQ (not really satisfied)
- more positive tail of the piecewie approx of Veq
- hessian of Lagrangia? Might need to expand r, in order to define variables as x0,r0,x1,r1
- extend terminal states by one M
	- should initial cost and terminal cost be computed over M init/final states instead of the first and last one?
	NB: it does not make much sense to extend it, and the reason is that, since the control action is fixed at that point, it would be like just increasing Np by 1. Instead, what might make a difference is on how many states to sum initial and terminal cost.



################# 6th meeting #################
Numerical Optimization. Algorithm 3.2 (pag. 48) describes generally the 2nd order method we would like to use



################# 7th meeting #################
- (V) normalize perturbation first action, so that I can specify a single weight good for both ramp and flow control 
- rate variability
	- (V) try lower constraint violation penalty, so that TTS matters more
	- (V) add control rate variation to RL stage cost
- try lower update frequencies
- try Gauss-Newton hessian approximation (still requires positive semidefinite modification) (just need to save dQ * dQ' as hessian and reduce as sum)
- backtracking
	- (V) cannot be done on the whole batch, just do it for the worst TD error
	- (V) the function is not really f. It is: f + v * sum(0, g), since our backtrack search is on a constrained box. The derivative is: df + v * dg * (sign() + 1), where v is larger than the inf norm of the multipliers of the original problem (for example, double of the last solution)
- baseline
	- (V) create a demand where the slope and height are randomly selected
	- (V) do a lot of multistart for the perfect information MPC, we have to be sure it is the best we can get
	- (V) baseline with perfect information (pars equal to true_pars, and no learnable cost weights) 
	- baseline with no learning
- scale learnable parameters



################# 8th meeting #################
- (V) add condition 3.6b to backtracking line search (Algorithm 3.1)
- (V) log how much hessian is modified at each iteration
- (V) compute duration of constraint violation in visualization plots
- (V) run GaussNetwon for 100-150 episodes
- (V) increase number of samples in backtracking, like the worst 10 td errors
- (see) use random demands: introduce training script (on a fixed seed) and validation seed (possibly without seed, which runs both the perfect information MPC and the MPC to validate)
- use rho crit tracking, and add a multiplicative term to the rho crit tracking cost term as:  (\rho^2)*||\rho - \rho_crit||^2 

- backlog
	- 0) reduce weight of control variability cost term (debatable)
	- 1) make train/valid split so that we have a good benchmark
		While learning can happen only once (i.e., train till convergence then save), validation should be done on N episodes for K times (so we get an average and std of the TTS reduction against the baseline)
	- 2) why not create a Gym like environment? Training should happen on K episodes, N times (e..g, on 10 episodes for 5 times = 50 episodes in total). In this way, at the end of the K episodes we can reset the gym. Validation should be done on K episodes for N times to get averages and stds.
		


############################ CODE FOR HESSIAN AND JACOBIAN, REMOVE IF NECESSARY

########## NMPC.m
obj.vars.w   = cell(1, obj.M * obj.Np + 1);
obj.vars.rho = cell(1, obj.M * obj.Np + 1);
obj.vars.v   = cell(1, obj.M * obj.Np + 1);
obj.vars.r   = cell(1, obj.Nc);
for k = 1:obj.M * obj.Np + 1
	obj.vars.w{k} = obj.opti.variable(n_orig, 1);
	obj.vars.rho{k} = obj.opti.variable(n_links, 1);
	obj.vars.v{k} = obj.opti.variable(n_links, 1);
	if mod(k - 1, obj.M) == 0 && k <= (obj.Nc - 1) * obj.M + 1
		obj.vars.r{ceil(k / obj.M)} = obj.opti.variable(n_ramps, 1);
	end
end
for name = fieldnames(obj.vars)'
	obj.vars.(name{1}) = horzcat(obj.vars.(name{1}){:});
end

########## main.m
tiledlayout(1, 2)
nexttile,
set(gca, 'XAxisLocation','top');
hold on, axis equal
ns = cumsum(structfun(@(x) numel(x), mpc.V.vars));
for i = 1:length(ns) - 1
    plot([ns(i), ns(i)], [1, mpc.V.opti.ng], 'k-.', 'LineWidth', 0.1)
end
spy(jacobian(mpc.V.opti.g, mpc.V.opti.x))
title('\nabla_x g(x,u)')
hold off,
axis([1, mpc.V.opti.nx, 1, mpc.V.opti.ng])

nexttile,
set(gca, 'XAxisLocation','top');
hold on, axis equal
ns = cumsum(structfun(@(x) numel(x), mpc.V.vars));
for i = 1:length(ns) - 1
    plot([ns(i), ns(i)], [1, ns(i)], 'k-.', 'LineWidth', 0.1)
    plot([1, ns(i)], [ns(i), ns(i)], 'k-.', 'LineWidth', 0.1)
end
spy(hessian(mpc.V.opti.f, mpc.V.opti.x))
title('\nabla_x^2 f(x)')
hold off,
axis([1, mpc.V.opti.nx, 1, mpc.V.opti.nx]), 
