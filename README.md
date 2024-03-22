# Reinforcement Learning with Model Predictive Control for Highway Ramp Metering

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/FilippoAiraldi/mpcrl-for-ramp-metering/blob/simulations/LICENSE)
![Python 3.11.4](https://img.shields.io/badge/python-3.11.4-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintainability](https://api.codeclimate.com/v1/badges/29611d19b6592f5d2ac2/maintainability)](https://codeclimate.com/github/FilippoAiraldi/mpcrl-for-ramp-metering/maintainability)

<div align="center">
  <img src="https://raw.githubusercontent.com/FilippoAiraldi/mpcrl-for-ramp-metering/simulations/resources/network.png" alt="network" height="200">
</div>

This repository contains the source code used to produce the results obtained in [Reinforcement Learning with Model Predictive Control for Highway Ramp Metering](https://arxiv.org/abs/2311.08820) submitted to [IEEE Transactions on Intelligent Transportation Systems](https://ieee-itss.org/pub/t-its/).

In this work, we propose to formulate the ramp metering control problem as a Markov Decision Process (MDP) and solve it using Reinforcement Learning (RL), where Model Predictive Control (MPC) acts as the function approximator. This combination allows us to leverage both the flexible, data-driven nature of RL and structured, model-based approach of MPC to come up with a learning-based control scheme that is able to tune its parametrisation automatically to enhance closed-loop performance.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{airaldi2023reinforcement,
  title = {Reinforcement Learning with Model Predictive Control for Highway Ramp Metering},
  author = {Airaldi, Filippo and De Schutter, Bart and Dabiri, Azita},
  journal={arXiv preprint arXiv:2311.08820},
  year = {2023},
  doi = {10.48550/ARXIV.2311.08820},
  url = {https://arxiv.org/abs/2311.08820}
}
```

---

## Installation

The code was created with `Python 3.11.4`. To access it, clone the repository

```bash
git clone https://github.com/FilippoAiraldi/mpcrl-for-ramp-metering.git
cd mpcrl-for-ramp-metering
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

In case you want to simulate also the additional agents contained in **`other_agents`** (explained below), please also install the corresponding requirements, e.g.,

```bash
pip install -r other_agents/requirements-pi-alinea.txt
```


### Structure

The repository code is structured in the following way (in alphabetical order)

- **`metanet`** contains the implementation of the training environment, which represent the traffic network benchmark, and is based on the METANET modelling framework, implemented in [sym-metanet](https://github.com/FilippoAiraldi/sym-metanet). The API follows the standarda OpenAI's `gym` style
- **`mpc`** contains the implementation of the MPC optimization scheme, which is based on [csnlp](https://github.com/FilippoAiraldi/casadi-nlp)
- **`other_agents`** contains implementations of the other agents compared against in the paper (non-learning MPC, PI-ALINEA, and DDPG).
- **`resouces`** contains media and other miscellaneous resources
- **`rl`** contains the implementation of the MPC-based RL agents, which are based on [mpcrl](https://github.com/FilippoAiraldi/mpc-reinforcement-learning)
- **`sim`** contains [lzma](https://docs.python.org/3/library/lzma.html)-compressed simulation results of different variants of the proposed approach
- **`util`** contains utility classes and functions for, e.g., constant, plotting, I/O, etc.
- **`launch.py`** launches simulations for different agents
- **`visualization.py`** visualizes the simulation results.

---

## Launching Simulations

Training and evaluation simulations can easily be launched via the command below. The provided arguments are set to reproduce the same main results found in the paper, assuming there are no discrepancies due to OS, CPU, etc.. For help about the implications of each different argument, run

```bash
python launch.py --help
```

In what follows, we provide the commands to reproduce the main results in the paper, for each type of agent. Note that the `runname` variable is used to name the output file, which will be saved under the filename `${runname}.xz`.

### MPC-based RL

Train with

```bash
python launch.py --agent-type=lstdq --gamma=0.98 --update-freq=240 --lr=1.0 --lr-decay=0.925 --max-update=0.3 --replaymem-size=2400 --replaymem-sample=0.5 --replaymem-sample-latest=0.5 --exp-chance=0.5 --exp-strength=0.025 --exp-decay=0.5 --agents=15 --episodes=80 --scenarios=2 --demands-type=random --sym-type=SX --seed=0 --verbose=1 --n-jobs=15 --runname=${runname}
```

### Non-learning MPC

Evaluate with

```bash
python launch.py --agent-type=nonlearning-mpc --gamma=0.98 --agents=15 --episodes=80 --scenarios=2 --demands-type=random --sym-type=SX --seed=0 --verbose=1 --n-jobs=15 --runname=${runname}
```

### PI-ALINEA

Evaluate with

```bash
python launch.py --agent-type=pi-alinea --Kp=32.07353865774536 --Ki=0.5419114131900662 --queue-management --agents=15 --episodes=80 --scenarios=2 --demands-type=random --sym-type=SX --seed=0 --verbose=1 --n-jobs=15 --runname=${runname}
```

The proportional and integral gains in PI-ALINEA can be fine-tuned by running

```bash
python other_agents/pi_alinea --tuned --n-trials=100 --agent=8
```

### DDPG

Train with

```bash
python launch.py --agent-type=ddpg --lr=1e-3 --gamma=0.98 --tau=1e-2 --batch-size=512 --buffer-size=200_000 --noise-std=0.3 --noise-decay-rate=5e-6 --devices=${your_devices} --agents=15 --episodes=80 --scenarios=2 --demands-type=random --sym-type=SX --seed=0 --verbose=1 --n-jobs=4 --runname=${runname}
```

---

## Visualization

To visualize simulation results, simply run

```bash
python visualization.py ${runname1}.xz ... ${runnameN}.xz --all
```

You can additionally pass `--paper`, which will cause the paper figures to be created (in this case, some of the simulation results' filepaths have been hardcoded for simplicity). For example, run the following to reproduce a part of the main figures in the paper

```bash
python visualization.py sims/sim_15_dynamics_a_rho_wo_track_higher_var.xz --all --paper
```

### Saved Results

Here we clarify the naming convention used for the saved simulation results, that can be found in the **`sims`** folder. Note that in each of the saved files, after decompression, you can find the arguments that were used to launch the simulation, as well as the simulation results themselves (which may differ from agent type to agent type).

Filenames always start with the name of the algorithm used, followed by the number of agents that were simulated. Then, additional information on each simulation can follow

- **MPC-based RL**: for these simulations (a.k.a., `lstdq`), we also report whether and which of the dynamics parameters (among `a`, `rho_crit`, and `v_free`) were allowed to be learnt, and if these were also used as tracking setpoints in the MPC objective (more details in the paper)
- **PI-ALINEA**: included is also whether the queue management strategy was enabled or not

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/FilippoAiraldi/mpcrl-for-ramp-metering/blob/simulations/LICENSE) file included with this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate [f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2023 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “mpcrl-for-ramp-metering” (Reinforcement Learning with Model Predictive Control for Highway Ramp Metering) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.
