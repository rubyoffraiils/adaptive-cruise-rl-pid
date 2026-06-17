import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from acc_env import AdaptiveCruiseControlEnv


env = AdaptiveCruiseControlEnv(scenario="hard")
model = PPO.load("ppo_acc_randomized")

obs, info = env.reset()

times = []
distances = []
errors = []
ego_speeds = []
lead_speeds = []
accelerations = []
jerks = []
collisions = 0

done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    times.append(info["time"])
    distances.append(info["distance"])
    errors.append(info["distance_error"])
    ego_speeds.append(info["ego_speed"])
    lead_speeds.append(info["lead_speed"])
    accelerations.append(info["ego_accel"])
    jerks.append(info["jerk"])

    if info["collision"]:
        collisions += 1


mean_error = np.mean(np.abs(errors))
min_distance = np.min(distances)
mean_jerk = np.mean(np.abs(jerks))
control_effort = np.sum(np.square(accelerations))

print("RL Evaluation Results")
print("---------------------")
print("Mean absolute distance error:", mean_error)
print("Minimum following distance:", min_distance)
print("Mean absolute jerk:", mean_jerk)
print("Control effort:", control_effort)
print("Collisions:", collisions)


plt.figure()
plt.plot(times, errors)
plt.axhline(0, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Distance error (m)")
plt.title("RL Adaptive Cruise Control - Distance Error")
plt.grid(True)
plt.savefig("rl_distance_error.png")
plt.show()

plt.figure()
plt.plot(times, distances, label="Actual distance")
plt.axhline(env.desired_distance, linestyle="--", label="Desired distance")
plt.xlabel("Time (s)")
plt.ylabel("Distance between cars (m)")
plt.title("RL Adaptive Cruise Control - Following Distance")
plt.legend()
plt.grid(True)
plt.savefig("rl_following_distance.png")
plt.show()

plt.figure()
plt.plot(times, ego_speeds, label="Ego speed")
plt.plot(times, lead_speeds, label="Lead speed")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("RL Adaptive Cruise Control - Vehicle Speeds")
plt.legend()
plt.grid(True)
plt.savefig("rl_vehicle_speeds.png")
plt.show()

plt.figure()
plt.plot(times, accelerations)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("RL Adaptive Cruise Control - Ego Acceleration")
plt.grid(True)
plt.savefig("rl_acceleration.png")
plt.show()
