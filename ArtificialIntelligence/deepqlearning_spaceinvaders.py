"""
Trains a Deep-Q Learning Agent to play Space Invaders using TensorFlow.
Source: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Space%20Invaders/DQN%20Atari%20Space%20Invaders.ipynb
"""
# IMPORTS ##################################################################################################

# Deep learning, math, environment
import tensorflow as tf
import numpy as np 
import retro

# Image processing, visualization
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque # Ordered collection with ends

# Misc.
import random
import warnings


# ENVIRONMENT SETUP ########################################################################################

game_name = 'SpaceInvaders-Atari2600'
env = retro.make(game=game_name)
print("***************** {} *****************".format(game_name))
print("Frame size: {}".format(env.observation_space))
print("Action size: {}\n".format(env.action_space.n))

# Here we create an hot encoded version of our actions
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())


# PRE-PROCESSING FUNCTIONS #################################################################################
"""
    preprocess_frame:
    Take a frame.
    Grayscale it
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    
    return preprocessed_frame
    
    """

# Function that shrinks and greyscales a frame
def preprocessFrame(frame):
    # Greyscale frame 
    gray = rgb2gray(frame)
    
    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12,4:-12]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    preprocessed_frame = transform.resize(cropped_frame, [110,84])
    
    return preprocessed_frame # 110x84x1 frame


stack_size  = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

# Function that stacks frames (to give Q-Agent a better sense of motion)
# Explanation why, if still curious: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
def stackFrames(stacked_frames, state, is_new_episode, stack_size=4):
    # Preprocess frame
    frame = preprocessFrame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames


# HYPERPARAMETERS DEFINED ###############################################################################

# Model hyperparameters
state_size = [110, 84, 4]        # 184x84 screen, with 4 frames (width, height, channels)
action_size = env.action_space.n # 8 possible actions
learning_rate = 0.00025          # Alpha (pretty self explanatory)

# Training hyperparameters
total_episodes = 50              # Number of training episodes
max_steps = 50000                # Max possible steps per episode
batch_size = 64                  # Batch size

# Exploration parameters hyperparameters
explore_start = 1.0              # Exploration probability high at start
explore_stop = 0.01              # Minimum exploration probability
decay_rate = 0.00001             # Rate the exploration probability decreases

# Q-Learning hyperparameters
gamma = 0.9                      # Reward discounting rate

# Memory hyperparameterse
pretrain_length = batch_size     # Number of experiences stored in memory
memory_size = 1000000            # Max number of experiences the memory keeps

# Pre-processing hyperparameters
stack_size = 4                   # Number of sequential images to use (gives idea of motion)

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

# TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


# DEEP Q-LEARNING NETWORK DEFINED BELOW ####################################################################

class DQLNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DeepQNetwork')


