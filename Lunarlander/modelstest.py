import gymnasium as gym
from stable_baselines3 import PPO
import os



env = gym.make('LunarLander-v3', render_mode='human') # continuous: LunarLanderContinuous-v2
env.reset()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

episodes = 5

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