import gymnasium as gym
from stable_baselines3 import DQN, PPO
import os





models_dir = "models/PPO"
model_path = f"{models_dir}/300000.zip"
env = gym.make('LunarLander-v3', render_mode='human') # continuous: LunarLanderContinuous-v2
env.reset()

model = PPO.load(model_path, env=env)


episodes = 1

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    #print(f"Episode {ep + 1}: Total Reward: {total_reward}")