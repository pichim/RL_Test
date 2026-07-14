import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make(
    "Pendulum-v1",
    render_mode="human",
)

model = SAC.load("sac_pendulum")

observation, info = env.reset(seed=2)

for step in range(400):
    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
