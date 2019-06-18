from DDPG_Agent import DDPGAgent
from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = DDPGAgent(seed=2)
agent.load()                                                    # Load network weights

env_info = env.reset(train_mode=False)[brain_name]            # reset the environment
states = env_info.vector_observations
agent.reset()
score = np.zeros(num_agents)

while True:
    actions = agent.act(states, add_noise=False)
    env_info = env.step(actions)[brain_name]               # send the action to the environment
    next_states = env_info.vector_observations               # get the next state
    rewards = env_info.rewards                               # get the reward
    dones = env_info.local_done                              # see if episode has finished

    agent.step(states, actions, rewards, next_states, dones)

    score += rewards                                         # update the score

    states = next_states                                     # roll over the state to next time step

    if np.any(dones):                                          # exit loop if episode finished
        break

print("Score : ", score)
