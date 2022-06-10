# Meta-gradient incentive design

This is the code for experiments in the paper [Adaptive Incentive Design with Multi-Agent Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/2112.10859), published at AAMAS 2022. Baselines are included.


## Setup

- `$ python3.6 -m venv <name of your venv>`
- `$ source <venv>/bin/activate`
- `$ pip install --upgrade pip`
- `$ git clone https://github.com/011235813/metagradient-incentive-design.git`
- `$ cd metagradient-incentive-design && pip install -e .`
- `$ pip install -r requirements.txt`
- Clone and `pip install` [Sequential Social Dilemma](https://github.com/011235813/sequential_social_dilemma_games), which is a fork from the [original](https://github.com/eugenevinitsky/sequential_social_dilemma_games) open-source implementation.
- Clone and `pip install` [AI Economist](https://github.com/011235813/ai-economist), which is a fork from the [original](https://github.com/salesforce/ai-economist)


## Navigation

* `alg/` - Implementation of MetaGrad and dual-RL baselines
* `configs/` - Experiment configuration files. Hyperparameters are specified here.
* `env/` - Implementation of 1) Escape Room game, 2) wrapper around the SSD environment, 3) wrapper around the Gather-Trade-Build scenario of the Foundation environment
* `results/` - Results of training will be stored in subfolders here. Each independent training run will create a subfolder that contains the final Tensorflow model, and reward log files. For example, training MetaGrad without curriculum on the 15x15 GTB map of Foundation would create `results/foundation/15x15_nocurr_m1` (depending on configurable strings in config files).
* `utils/` - Utility methods


## Examples

### Train MetaGrad on Escape Room

* Set config values in `configs/config_er_pg.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_er.py pg`.

### Train MetaGrad on Cleanup

* Set config values in `configs/config_ssd.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_ssd.py ac`.

### Train MetaGrad on GTB

Training without curriculum
* Set config values in `configs/config_foundation_ppo.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_foundation.py ppo`.

To enable curriculum learning, i.e. use a policy pretrained on a free-market scenario
* Set config values in `configs/config_foundation_ppo_curriculum.py`
* The pretrained model is located at `results/foundation/15x15_phase1_free_market/model.ckpt`
* `cd` into the `alg` folder
* Execute training script `$ python train_foundation.py curr`.


## Citation

<pre>
@inproceedings{yang2022adaptive,
  title={Adaptive Incentive Design with Multi-Agent Meta-Gradient Reinforcement Learning},
  author={Yang, Jiachen and Wang, Ethan and Trivedi, Rakshit and Zhao, Tuo and Zha, Hongyuan},
  booktitle={Proceedings of the 21st International Conference on Autonomous Agents and MultiAgent Systems},
  pages={1436--1445},
  year={2022}
}
</pre>

## License

See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT
