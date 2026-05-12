from snakeenv import SnekEnv


env = SnekEnv()
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		step_result = env.step(random_action)
		obs, reward, done, info = step_result[:4]
		print('reward',reward)