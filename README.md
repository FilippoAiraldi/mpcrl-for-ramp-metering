# Reinforcement Learning with Model Predictive Control for Highway Ramp Metering

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/FilippoAiraldi/mpcrl-for-ramp-metering/blob/simulations/LICENSE)
![Python 3.11.4](https://img.shields.io/badge/python-3.11.4-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- [![Maintainability](https://api.codeclimate.com/v1/badges/0f8d97e0b178bb04f0af/maintainability)](https://codeclimate.com/github/FilippoAiraldi/learning-safety-in-mpc-based-rl/maintainability) -->

<div align="center">
  <img src="https://raw.githubusercontent.com/FilippoAiraldi/mpcrl-for-ramp-metering/simulations/resources/network.png" alt="network" height="200">
</div>

This repository contains the source code used to produce the results obtained in [our paper](TODO) submitted to TODO.

In this work, we propose to formulate the ramp metering control problem as a Markov Decision Process (MDP) and solve it using Reinforcement Learning (RL), where Model Predictive Control (MPC) acts as the function approximator. This combination allows us to leverage both the flexible, data-driven nature of RL and structured, model-based approach of MPC to come up with a learning-based control scheme that is able to tune its parametrisation automatically to enhance closed-loop performance.

If you find the paper or this repository helpful in your publications, please consider citing it.

<!-- ```latex
article{airaldi2022learning,
  author = {Airaldi, Filippo and De Schutter, Bart and Dabiri, Azita},
  title = {Reinforcement Learning with Model Predictive Control for Highway Ramp Metering},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2211.01860},
  year = {2023},
  doi = {10.48550/ARXIV.2211.01860},
  url = {https://arxiv.org/abs/2211.01860}
}
``` -->

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

### Structure

The repository code is structured in the following way

- **`metanet`** contains the implementation of the training environment, which represent the traffic network benchmark, and is based on the METANET modelling framework. The implementation follows the standarda OpenAI's `gym` style
- **`mpc`** contains the implementation of the MPC optimization scheme, which is based on [csnlp](https://github.com/FilippoAiraldi/casadi-nlp)
- **`resouces`** contains media and other miscellaneous resources
- **`rl`** contains the implementation of the RL agents, which are based on [mpcrl](https://github.com/FilippoAiraldi/mpc-reinforcement-learning)
- **`sim`** contains [lzma](https://docs.python.org/3/library/lzma.html)-compressed simulation results of different variants of the proposed approach
- **`util`** contains utility classes and functions for, e.g., constant, plotting, I/O, etc.
- **`train.py`** launches simulations for agents
- **`visualization.py`** visualizes the simulation results.

---

## Training

Training simulations can easily be launched via the command below. The provided arguments are set to reproduce the same main results found in the paper, assuming there are no discrepancies due to OS, CPU, etc..

```bash
python -u train.py --agent-type=lstdq --gamma=0.98 --update-freq=240 --lr=1.0 --lr-decay=0.925 --max-update=0.3 --replaymem-size=2400 --replaymem-sample=0.5 --replaymem-sample-latest=0.5 --exp-chance=0.5 --exp-strength=0.025 --exp-decay=0.5 --agents=15 --episodes=80 --scenarios=2 --demands-type=random --sym_type=SX --seed=0 --verbose=1 --n_jobs=15 --runname=${runname}
```

Results will be saved under the filename `${runname}.xz`. For help about the implications of each different argument, run

```bash
python train.py --help
```

---

## Visualization

To visualize simulation results, simply run

```bash
python visualization.py ${runname1}.xz ... ${runnameN}.xz --all
```

You can additionally pass `--paper`, which will cause the paper figures to be created (in this case, the simulation results filepaths are hardcoded). For example, run the following to reproduce the main figures in the paper

```bash
python visualization.py sims/sim_15_dynamics_a_rho_wo_track_higher_var.xz --all --paper
```

## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/FilippoAiraldi/mpcrl-for-ramp-metering/blob/simulations/LICENSE) file included with this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate [f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2023 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “mpcrl-for-ramp-metering” (Reinforcement Learning with Model Predictive Control for Highway Ramp Metering) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.
