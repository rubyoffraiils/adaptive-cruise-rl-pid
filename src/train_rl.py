"""
Single-run PPO trainer. Saves the model to rl_runs/run3/.
For seed sweeps use train_seeds.py; for style variants use train_styles.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from acc_env import AdaptiveCruiseControlEnv

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "rl_runs", "run3")
os.makedirs(OUT_DIR, exist_ok=True)

# train on randomized so the policy learns to react to distance/speed cues
# rather than memorising a fixed timing sequence
env = AdaptiveCruiseControlEnv(scenario="randomized")
check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,   # steps per rollout before each gradient update
    batch_size=128,
    gamma=0.99,     # discount factor — how much future rewards matter
)
model.learn(total_timesteps=1_500_000)

out_path = os.path.join(OUT_DIR, "model")
model.save(out_path)
print(f"Model saved → {out_path}.zip")
