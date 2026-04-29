import gymnasium as gym 
# Create the environment
env = gym.make('LunarLander-v3', render_mode='human') # continuous: LunarLanderContinuous-v2 
# required before you can step the environment 
env.reset() 
# sample action: 
for step in range(50):
    env.render()
    #take random action
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    print(f"Reward: {reward}, Done: {done}")

env.close()