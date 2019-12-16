# Solving Lunar Lander Envoirnment using Deep Reinforcement Learning
### By Neel Bhave and Winston Moh Tanghoho

We solved the [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) envoirnment provided by OpenAI using Deep Reinforcement Learning.

We used [Keras-RL](https://github.com/keras-rl/keras-rl) and [PyTorch](https://pytorch.org/) to accomplish this. 

This repo contains the following files -

1. LunarLander_v2.py - This file contains the training as well as testing code for the envoirnment. Refer to the comments inside the file to know how to run it in test or train mode. This file trains the agent using Keras.

2. Lunar_Lander_DQN_Pytorch.py - This file contains the code for training the DQN agent using Pytorch.

3. LunarLanderDQN.ipynb - This file is a Python Notebook that was used to train the agent on Google Colab. This file uses the similar code as in LunarLander_v2.py to train the model.

4. Trained Weights Folder - This folder contains the trained weights that can be used to test the agent in test mode in the LunarLander_v2.py. The name of the file should be changed on line 98 in that file to the name of the weight file that is to be loaded.

5. FAI_Project_Report_Final.pdf - This is the final report for this project and explains how Deep RL was used to train a agent with optimal policy for this envoirnment and also the shortcomings of other RL and Deep RL methods are discussed.

### Installation Instructions -

* Install the depedencies in             requirements.txt using - 
  ```
  pip install -r requirements.txt
  ```
  You might want to create a virtual envoirnments as the project requires large number of packages to be installed.

