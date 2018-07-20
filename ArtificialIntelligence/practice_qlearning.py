"""
Program that learns to navigate the OpenAI gym's 'Frozen Lake' environment,
using a Q-learning table.
"""

# IMPORTS, SETUP, HYPERPARAMETERS DEFINED ######################################################################
import numpy as np
import gym, random
import time, os

# Set up our environment, get the action and state sizes
env = gym.make('FrozenLake-v0')
action_size = env.action_space.n
state_size = env.observation_space.n

# Initialize our Q-table
qtable = np.zeros((state_size, action_size))
print(qtable)

# Hyper parameters defined here
total_episodes = 5000
learning_rate = 0.8
max_steps = 99
gamma = 0.95            # Discount rate- weighs future reward
# EXPLORATION PARAMETERS BELOW
epsilon = 1.0           # Exploration rate
max_epsilon = 1.0       # Highest exploration rate (used at start)
min_epsilon = 0.01      # Lowest exploration rate (once it learns, it'll go for the gold)
decay_rate = 0.01       # Exponential decay rate for exploration probability


# MAIN Q-LEARNING ALGORITHM IMPLENTED BELOW #######################################################################

# List of rewards
rewards = []

# For life, or until learning is stopped....
for episode in range(total_episodes):
    # Reset the environment from previous round
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0.0

    # Start walking through the environment
    for step in range(max_steps):
        explore_exploit_tradeoff = random.uniform(0, 1)   # Initialize random exploration/exploitation tradeoff
        # If the tradeoff is greater than epsilon, we exploit- get that reward bby
        if explore_exploit_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        # Otherwise, we go into exploring mode- try to get the lay of the land
        else:
            action = env.action_space.sample()

        # Take the action, get the reward/outcome
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + (gamma * np.max(qtable[new_state, :])) - qtable[state, action])
        total_rewards += reward

        # New state becomes current state, or break if we died and are done; increment episode
        state = new_state
        if done: break

    # Reduce epsilon, to explore less
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    rewards.append(total_rewards)

    os.system('cls' if os.name == 'nt' else 'clear')
    print ("Score after episode: " + str(episode) + ": " +  str(total_rewards))
    print(qtable)
    print("\n")


# ACTUALLY PLAYS FROZEN LAKE, POST TRAINING! #####################################################################
print("TOTAL REWARDS FROM TRAINING: " +  str(sum(rewards)))
env.reset()

for  episode in range(10):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0
    print("**************************************************************************************")
    print("EPISODE ", episode)
    print("**************************************************************************************")
    time.sleep(2)

    for step in range(max_steps):
        env.render()
   
        action = np.argmax(qtable[state,:]) # Should get the best action given the state
        
        new_state, reward, done, info = env.step(action)
        
        total_reward += reward

        if done: break
        state = new_state

        time.sleep(0.1)
        os.system('cls' if os.name == 'nt' else 'clear')

    print("**********************************")
    print("REWARD THIS ROUND: ", total_reward)
    print("**********************************\n\n")
    time.sleep(1)
    os.system('cls' if os.name == 'nt' else 'clear')


env.close()


"""
# Challenge program with Edwin (unrelated to AI)
x = input("Enter your number: ")
x = int(x)
num = 0
for i in range(1, x+1):
    num += (i * (10**(x-i)))

print("Output: {}".format(num))

"""