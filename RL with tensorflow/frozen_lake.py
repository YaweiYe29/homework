import gym.spaces
import numpy as np

env = gym.make('FrozenLake-v0')
s = env.reset()
print(s)

env.render()

print(env.action_space)         # number of actions
print(env.observation_space)    # number of states

print("Number of actions: ",env.action_space.n)
print("Number of states: ",env.observation_space.n)

# Epsilon Greedy
def epsilon_greedy(Q, state):
    epsilon = 0.3
    p = np.random.uniform(low=0, high=1)
    if p > epsilon:
        return np.argmax(Q[s,:])
    else:
        return env.action_space.sample()

if __name__ =="__main__":

    # Q-learning Implementation
    Q = np.zeros([env.observation_space.n, env.action_space.n])
