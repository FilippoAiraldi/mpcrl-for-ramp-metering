from operator import neg
from typing import Any, Literal

import numpy as np
from gymnasium import Env, ObservationWrapper, spaces
from gymnasium.wrappers import TransformReward
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from metanet import HighwayTrafficEnv
from util import EnvConstants as EC
from util import MpcRlConstants as MRC
from util.constants import STEPS_PER_SCENARIO


class ReplayBufferWithLookAhead(ReplayBuffer):
    """Replay buffer with look-ahead, i.e., with N-step return to estimated the target,
    for off-policy algorithms."""

    def __init__(self, *args: Any, lookahead: int, gamma: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lookahead = lookahead
        self.gamma = gamma
        self.weights = np.power(gamma, np.arange(lookahead), dtype=self.rewards.dtype)

    def _get_samples(
        self, inds: np.ndarray, env: VecNormalize | None = None
    ) -> ReplayBufferSamples:
        if self.lookahead <= 1:
            return super()._get_samples(inds, env)

        # grab some constants
        batch_size = len(inds)
        pos = self.pos
        lookahead = self.lookahead
        size = self.buffer_size
        weights = self.weights
        env_inds = np.random.randint(0, high=self.n_envs, size=(batch_size,))

        # first, sample state-action pairs, as they do not depend on the lookahead
        obs = self.observations[inds, env_inds, :]
        act = self.actions[inds, env_inds, :]

        # then, it's time to sample the rewards, dones, timeouts and next states.
        # Starting from inds, we increment it by one along the lookahead, and accumulate
        # the rewards. We stop when we reach done or timeout, or when we reach the max
        # index. Computations are carried out in vectorized form along the batch dim
        cumreward = np.zeros((batch_size, 1), dtype=self.rewards.dtype)
        max_inds = np.minimum(inds + lookahead, pos + (inds >= pos) * size)
        should_be_incremented = np.arange(batch_size, dtype=int)
        for w in weights:
            inds_ = inds[should_be_incremented] % size
            env_inds_ = env_inds[should_be_incremented]
            cumreward[should_be_incremented, 0] += w * self.rewards[inds_, env_inds_]

            # remove from inds those that are done, timeouted or reached the max index
            mask = np.logical_and(
                np.logical_and(
                    self.dones[inds_, env_inds_] < 1.0,
                    self.timeouts[inds_, env_inds_] < 1.0,
                ),
                inds[should_be_incremented] != max_inds[should_be_incremented] - 1,
            )
            should_be_incremented = should_be_incremented[mask]
            if should_be_incremented.size == 0:
                break

            # increment the corresponding indices
            inds[should_be_incremented] += 1

        inds %= size
        next_obs = self.next_observations[inds, env_inds]
        done = self.dones[inds, env_inds].reshape(-1, 1)
        timeout = self.timeouts[inds, env_inds].reshape(-1, 1)

        # normalize, exclude dones due to timeouts, and return
        data = (
            self._normalize_obs(obs, env),
            act,
            self._normalize_obs(next_obs, env),
            done * (1.0 - timeout),
            self._normalize_reward(cumreward, env),
        )
        return ReplayBufferSamples(*map(self.to_torch, data))


class DecayNoiseCallback(BaseCallback):
    """Callback that decays the action noise std by the given rate at each step."""

    def __init__(
        self, noise: OrnsteinUhlenbeckActionNoise, decay_rate: float, verbose: int = 0
    ) -> None:
        assert hasattr(noise, "_sigma"), "Action noise must have a '_sigma' attribute."
        super().__init__(verbose)
        self.noise = noise
        self.decay_rate = 1.0 - decay_rate

    def _on_step(self) -> bool:
        self.noise._sigma *= self.decay_rate
        return True


class AugmentedObservationWrapper(ObservationWrapper):
    """Wrapper that augments the observation with disturbances and previous input."""

    def __init__(self, env: HighwayTrafficEnv) -> None:
        super().__init__(env)
        na = env.na
        nd = env.nd
        obs = env.observation_space
        self.observation_space = spaces.Box(
            np.concatenate((obs.low, np.zeros(nd * EC.steps + na))),
            np.concatenate((obs.high, np.full(nd * EC.steps, np.inf), np.ones(na))),
            dtype=obs.dtype,
            seed=obs.np_random,
        )

    def observation(self, state: np.ndarray) -> np.ndarray:
        env: HighwayTrafficEnv = self.env
        new_state = np.concatenate(
            (state, env.demand[env.demand.t].T, env.last_action), axis=None
        )
        assert self.observation_space.contains(new_state), "Invalid observation."
        return new_state


def make_env(
    gamma: float,
    scenarios: int,
    demands_type: Literal["constant", "random"],
    sym_type: Literal["SX", "MX"],
    evaluation: bool = False,
    seed: int | None = None,
) -> VecNormalize:
    """Creates and appropriately wraps the traffic env for training or evaluation."""
    MRC.normalization["a"] = 1.0  # since here we control rate (and not flow of O2)
    env = HighwayTrafficEnv.wrapped(
        control_O2_rate=True,  # control rate rather than flow
        demands_type=demands_type,
        sym_type=sym_type,
        n_scenarios=scenarios,
        monitor_deques_size=None if evaluation else 0,  # do not record in training
    )
    env = AugmentedObservationWrapper(env)
    env = TransformReward(env, neg)
    env = Monitor(env)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize(
        venv, not evaluation, clip_obs=np.inf, clip_reward=np.inf, gamma=gamma
    )
    venv.seed(seed)
    return venv


def train_ddpg(
    episodes: int,
    scenarios: int,
    learning_rate: float,
    batch_size: int,
    buffer_size: int,
    tau: float,
    gamma: float,
    noise_std: float,
    noise_decay_rate: float,
    device: str,
    demands_type: Literal["constant", "random"],
    sym_type: Literal["SX", "MX"],
    seed: int,
    verbose: int,
) -> Env:
    """Trains a DDPG agent on the traffic environment.

    Parameters
    ----------
    episodes : int
        Number of episodes to train the agent for.
    scenarios : int
        Number of demands' scenarios per episode.
    learning_rate : float
        The learning rate of the RL algorithm.
    batch_size : int
        Size of mini-batches sampled from the replay buffer when updating.
    buffer_size : int
        Size of the whole replay buffer.
    tau : float
        The soft update coefficient (target = tau * target + (1 - tau) * current).
    gamma : float
        Discount factor.
    noise_std : float
        Standard deviation of the action noise (Ornstein-Uhlenbeck).
    noise_decay_rate : float
        Decay rate of the std of the action noise.
    device : "auto", or "cpu", or "cuda:0", etc.
        Device to use for training.
    demands_type : {"constant", "random"}
        Type of demands affecting the network.
    sym_type : {"SX", "MX"}
        The type of casadi symbols to use during the simulation.
    seed : int
        RNG seed for the simulation.
    verbose : {0, 1, 2, 3}
        Level of verbosity of the agent's logger.

    Returns
    -------
    Env
        The wrapped instance of the traffic environment.
    """
    set_random_seed(seed, using_cuda="cuda" in device)

    # create the model
    lookahead = MRC.prediction_horizon
    env = make_env(gamma, scenarios, demands_type, sym_type, seed=seed)
    na = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        np.zeros(na), np.full(na, noise_std), dt=1.0
    )
    model = TD3(  # use TD3 because it allows for delaying the policy updates
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma**lookahead,  # to account for lookahead target estimation
        train_freq=(1, "episode"),
        gradient_steps=-1,
        action_noise=action_noise,
        replay_buffer_class=ReplayBufferWithLookAhead,
        replay_buffer_kwargs={"lookahead": lookahead, "gamma": gamma},
        policy_delay=lookahead,  # keep only this trick
        policy_kwargs={"n_critics": 1, "net_arch": [256, 256]},  # remove twin critic
        target_noise_clip=0.0,  # remove additional noise trick
        verbose=verbose,
        seed=seed,
        device=device,
    )

    # create training callbacks - NOTE: unfortunately, I found no way of seeding the env
    # at each evaluation step, so I cannot guarantee complete reproducibility. Evaluate
    # every time after each of the first 80 episodes, then every 10 episodes
    STEPS_PER_EP = STEPS_PER_SCENARIO * scenarios - 1
    eval_env = make_env(
        gamma, scenarios, demands_type, sym_type, evaluation=True, seed=seed
    )
    eval_cb = EvalCallback(
        eval_env=eval_env, n_eval_episodes=1, eval_freq=STEPS_PER_EP, verbose=verbose
    )
    decay_action_noise_cb = DecayNoiseCallback(action_noise, noise_decay_rate)
    callback = [decay_action_noise_cb, eval_cb]

    # launch the training
    total_timesteps = STEPS_PER_EP * episodes
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=callback)
    return eval_env.venv.envs[0].env.env.env  # ugly, but we only want the MonitorInfos
