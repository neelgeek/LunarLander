# Import gym module.
import gym
import argparse
import numpy as np

from keras.models import Sequential,InputLayer
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.sarsa import SARSAAgent
from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent

from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

train=True
steps=10000
# Command Line Args Instructions:
#     Use the argument -m or --mode to specify the mode
#     If you want to train the model,pass "train"
#     If you want to test the model,pass "test"
#     By default the option is train
#     E.g. python LunarLander-v2.py -m test

#     Use the argument -s or --steps to specify the number of steps
#     E.g. python LunarLander-v2.py -m train -s 10000      



parser = argparse.ArgumentParser()
parser.add_argument("--mode","-m",help="Specify if the mode is test or train")
parser.add_argument("--steps","-s",help="Specifies the number of training steps")
args = parser.parse_args()

if args.mode=="test":
    train=False

if args.steps:
    steps=int(args.steps)



# Landing pad is always at coordinates (0,0). First two numbers in state vector.
#
# Reward to move from top screen to landing pad at zero speed is about 100..140 points. 
#
# Move away and lose reward.
#
# Episode is over if the lander crashes or comes to rest. Receiving additional -100 or +100 points.
#
# Each leg ground contact is +10
# Firing main engine is -0.3 points each frame.
#
# Solved is 200 points.
# 
# 4 discrete actions available: - do nothing, 
#				- fire left orientation engine, 
#				- fire main engine,
#				- fire right orientation engine.

# state space has 8 features
# state_space = [posx, posy, x_velocity, y_velocity, 
#               lander_angle, lander_angular_velocity, leg0_touches_ground, leg1_touches_ground]
# To run program $ python3 LunarLander-v2.py

# create environment.
env_name = "BipedalWalker-v2"
env = gym.make(env_name)



env.seed(1000)
nb_actions = env.action_space.n  # get number of actions in Lunar Lander

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
# Need another layer for predicted network. Also need to perform Experience replay.
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=10000, window_length=1)
agent = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
# agent=DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,target_model_update=1e-2, policy=policy)
agent.compile(Adam(lr=1e-4), metrics=['mae'])

if train:
    # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot. 
    agent.fit(env, nb_steps=steps, visualize=False)
    agent.save_weights(env_name+' '.join(map(str, agent.metrics_names))+".weights",overwrite=True)
else:
    agent.load_weights("LunarLander-v2loss mean_absolute_error mean_q(DQN).weights")
    
agent.test(env, nb_episodes=10, visualize=True)

