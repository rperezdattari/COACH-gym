import gym
import time
from linear_RBFs import LinearRBFs
import numpy as np

# Set configuration parameters
render = True
save_trained_parameters = True
max_num_of_episodes = 200
max_time_steps_episode = 1000

# Create environment
env = gym.make('Continuous-CartPole-COACH-v1')

# Initialize agent
agent = LinearRBFs(load_trained_parameters=False)

# Initialize obtained reward
reward = 0

# Iterate over the maximum number of episodes
for i_episode in range(max_num_of_episodes):
    observation = env.reset()  # If the environment is reset, the first observation is given
    agent.new_episode()  # Reset episode variables

    # Iterate over all episodes
    print('Starting episode number', i_episode)
    for t in range(max_time_steps_episode):
        if render:
            env.render()  # Make the environment visible
        action = agent.action(observation)
        observation, h, done, info = env.step(action)  # Receive an observation and reward after action
        reward += h[1]  # h[0]: human feedback; h[1]: reward value

        if h[0] != 0:  # If feedback is given
            agent.update(h[0], observation)

        if done:  # If the episode is finished
            if save_trained_parameters:
                agent.save_params()

            print('episode reward:', reward)
            reward = 0
            time.sleep(1)
            break
