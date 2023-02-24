# Highway Traffic Control with MPC-based RL
Repository for an MPC-based RL framework to tackle the learning control problem for a highway traffic control application. 

---
## Installation
The code is written in [MATLAB R2021b](https://www.mathworks.com/products/new_products/release2021b.html). For automatic differentiation and MPC optimization, the [CasADi](https://web.casadi.org/) framework (v3.5.5) is employed.

---
## Problem and Methods

### Definition
The task is to find the optimal control policy for the following small highway stretch. 
![Network Image](imgs/network.png)
The available control action is the incoming flow entering the highway from the ramp (via ramp metering). The goal is to control the flow in such a way to minimize the average time spent traversing the network, while avoiding constraint violation on the length of the ramp queue.

### Methodology
To learn this policy, an MPC controller is used as policy provider (i.e., function approximation), while an RL agent is tasked with tuning the controller's parameters to achieve optimal performance. 

### Structure
The repository is structured in the following way
- `imgs`: contains images related to the application.
- `src`: this folder contains the actual Matlab files used in solving the problem at hand
    - `+METANET`: contains functions and class to simulate the highway environment according to the well-known METANET modeling framework
    - `+MPC`: contains functions and class to instantiate and run a nonlinear MPC controller 
    - `+RL`: contains functions and classes to instantiate different RL agents
    - `+util`: contains all the utility functions for, e.g., logging, plotting during training, etc..
    - `trainDPG.m`: <span style="color:red">TODO</span>
    - `trainQL.m`: trains a Q learning agent 
    - `valid.m`: runs a validation session for multiple agents

---
## License
The repository is provided under the GNU General Public License. See the LICENSE file included with this repository. 

--
## Citation
The work provided in this repository can be cited as TODO

---
## Author
Filippo Airaldi [filippoairaldi@gmail.com | f.airaldi@tudelft.nl]

Copyright (c) 2023 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “TODO” (TODO) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.
