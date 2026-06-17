from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from acc_env import AdaptiveCruiseControlEnv


# Train on the randomized scenario so the policy learns to react to
# distance/speed cues rather than memorising a fixed timing sequence.
env = AdaptiveCruiseControlEnv(scenario="randomized")

# Check that the environment follows Gymnasium rules
check_env(env, warn=True)

# Create PPO model.
# n_steps=2048 gives PPO a longer rollout window — important for ACC because
# the consequences of an action (closing gap, overshooting) unfold over several
# seconds, not just one step.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

# 500k steps — 5x run 1 — gives the policy enough episodes across enough
# random profiles to learn a general reactive strategy.
model.learn(total_timesteps=500_000)

# Save trained model
model.save("ppo_acc_randomized")

print("Training complete. Model saved as ppo_acc_randomized.")
