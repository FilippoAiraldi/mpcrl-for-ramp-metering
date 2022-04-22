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



## NOTES
1. do not use solutions where casadi has not converged to optimal

2. remember to normalize

3. MPC
	- initial
		affine term (EMPC) in states (can be also a parameter to learn)

	- stage
		quadratic in density with tracking of rho_crit (may be fixed to the starting known wrong value)
		quadratic in speed with tracking alpha (can it be v_free itself or is it too high?)
		
	- terminal
		quadratic in density with tracking of rho_crit

   RL
	same stage cost as MPC + constraint violation penalty (weights are fixed)
