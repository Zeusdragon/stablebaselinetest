import gymnasium as gym
from stable_baselines3 import DQN, PPO
from sb3_contrib import QRDQN
import os


def get_model_step(episode_num):
    """
    Logarithmische Verteilung der Modellschritte:
    - Episoden 1-10: 10000, 20000, ..., 100000 (lineare Schritte à 10k)
    - Episoden 11+: exponentielle Schritte (150k, 200k, 300k, 400k, 500k)
    """
    if episode_num <= 10:
        return episode_num * 10000
    else:
        # Nach 10 Episoden: 150k, 200k, 300k, 400k, 500k
        steps = [150000, 200000, 300000, 400000, 500000]
        idx = min(episode_num - 11, len(steps) - 1)
        return steps[idx]




models_dir = "models/QRDQN"
env = gym.make('LunarLander-v3', render_mode='human')  # continuous: LunarLanderContinuous-v2

episodes = 15

for ep in range(episodes):
    # Lade das entsprechende Modell für diese Episode
    model_step = get_model_step(ep + 1)
    model_path = f"{models_dir}/{model_step}.zip"
    
    print(f"\n=== Episode {ep + 1} ===")
    print(f"Lade Modell: {model_path}")
    
    env.reset()
    model = QRDQN.load(model_path, env=env)
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    
    print(f"Episode {ep + 1}: Total Reward: {total_reward}")