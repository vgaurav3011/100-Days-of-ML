import gym
import numpy as np
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

for episode in range(num_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        # Exploration-Exploitation Tradeoff

        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])

        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        q_table[state, action] = q_table[state, action] * (1-learning_rate)+ \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        rewards_current_episode += reward

        if done==True:
            break
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000

print("Average reward per thousand episodes")

for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count+=1000

print(q_table)

for episode in range(3):
    state = env.reset()
    done = False
    print("EPISODE ", episode+1, "\n\n\n\n\n")
    time.sleep(1)

for step in range(max_steps_per_episode):
    clear_output(wait=True)
    env.render()
    time.sleep(0.3)

    action = np.argmax(q_table[state,:])
    new_state, reward, done, info = env.step(action)

    if done:
        clear_output(wait=True)
        env.render()
        if reward == 1:
            print("You reached the goal!!")
            time.sleep(3)

        else:
            print("You fell through a hole")
            time.sleep(3)
        clear_output(wait=True)
        break
    state = new_state
env.close()