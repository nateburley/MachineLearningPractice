"""
Program that uses a convolutional neural network implemented in TensorFlow to 
play doom, using the following Deep Q-Learning Algorithm:

Initialize Doom Environment E
Initialize replay Memory M with capacity N (= finite capacity)
Initialize the DQN weights w
for episode in max_episode:
    s = Environment state
    for steps in max_steps:
         Choose action a from state s using epsilon greedy.
         Take action a, get r (reward) and s' (next state)
         Store experience tuple <s, a, r, s'> in M
         s = s' (state = new_state)
         
         Get random minibatch of exp tuples from M
         Set Q_target = reward(s,a) +  γmaxQ(s')
         Update w =  α(Q_target - Q_value) *  ∇w Q_value
"""
# IMPORTS ##################################################################################################

import tensorflow as tf         # Deep Learning library
import numpy as np              # Handle matrices
from vizdoom import *           # Doom Environment

import random                   # Handling random number generation
import time                     # Handling time calculation
from skimage import transform   # Help us to preprocess the frames

from collections import deque   # Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings                 # This ignore all the warning messages from skimage
warnings.filterwarnings('ignore')


# ENVIRONMENT SETUP ########################################################################################

# Function that creates the environment
def createEnvironment():
    game = vizdoom.DoomGame()

    # Load basic configuration
    game.load_config('basic.cfg') 

    # Load correct scenario (basic scenario here)
    game.set_doom_scenario_path('basic.wad')

    game.init()

    # Possible actions
    left = [1,0,0]
    right = [0,1,0]
    shoot = [0,0,1]
    possible_actions = [left, right, shoot]

    return game, possible_actions


# Function that tests the environment
def testEnvironment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print ("\treward:", reward)
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()


# PREPROCESSING OF IMAGES/FRAMES ###########################################################################
"""
Preprocessing is an important step, because we want to reduce the complexity of our states to reduce 
the computation time needed for training. 

Our steps:

-Grayscale each of our frames (because color does not add important information ). But this is already 
done by the config file.
-Crop the screen (in our case we remove the roof because it contains no information)
-Normalize pixel values
-Finally, we resize the preprocessed frame
"""

