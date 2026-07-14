import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("Pendulum-v1")

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    seed=1,
)

model.learn(total_timesteps=10000)

model.save("sac_pendulum")

env.close()

print("Training completed and model saved.")
