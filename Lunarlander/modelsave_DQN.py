import gymnasium as gym
from stable_baselines3 import DQN
import os

models_dir = "models/DQN"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs"

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('LunarLander-v3') # continuous: LunarLanderContinuous-v2
env.reset()

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
for i in range(30):
    iters +=1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")