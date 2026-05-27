from __future__ import annotations

import importlib.util

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TimeCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        window_size, num_features = observation_space.shape
        self.window_size = int(window_size)
        self.num_features = int(num_features)
        self.cnn = nn.Sequential(
            nn.Conv1d(self.num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        cnn_out = 64 * max(1, self.window_size // 4)
        self.linear = nn.Sequential(nn.Linear(cnn_out, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.transpose(1, 2)
        return self.linear(self.cnn(x))


def build_model(policy_type: str, env, run_dir, seed: int, total_timesteps: int, params: dict | None = None):
    params = params or {}
    base_kwargs = dict(
        env=env,
        verbose=1,
        seed=seed,
        n_steps=params.get("n_steps", 4096),
        batch_size=params.get("batch_size", 512),
        ent_coef=params.get("ent_coef", 0.03),
        learning_rate=params.get("learning_rate", 3e-4),
    )
    if importlib.util.find_spec("tensorboard") is not None:
        base_kwargs["tensorboard_log"] = str(run_dir / "tensorboard")

    if policy_type == "cnn1d":
        base_kwargs["policy_kwargs"] = dict(
            features_extractor_class=TimeCNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        )
        return PPO("MlpPolicy", **base_kwargs)

    if policy_type == "recurrent_lstm":
        if importlib.util.find_spec("sb3_contrib") is None:
            raise ImportError("recurrent_lstm requires sb3-contrib. Install it or choose mlp/cnn1d.")
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO("MlpLstmPolicy", **base_kwargs)

    base_kwargs["policy_kwargs"] = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    return PPO("MlpPolicy", **base_kwargs)
