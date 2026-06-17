from acc_env import AdaptiveCruiseControlEnv

env = AdaptiveCruiseControlEnv(scenario="default")

obs, info = env.reset()
print("Initial observation:", obs)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print("\nStep:", i)
    print("Action:", action)
    print("Observation:", obs)
    print("Reward:", reward)
    print("Distance:", info["distance"])
    print("Collision:", info["collision"])

    if terminated or truncated:
        break
