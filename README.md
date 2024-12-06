All code found here can be opened in VSCode.  User needs to appropriately install the following: "pip install kaggle-environments", "pip install matplotlib", "pip install stable-baselines3" (On personal terminal or in notebook). 

The code found in Reinforcement Learning gent.py is inspired by Alexis Cook's Deep Reinforcement Learning tutorial. (https://www.kaggle.com/code/alexisbcook/deep-reinforcement-learning) This is the file we used to train our agent by developing a model using the PPO algorithm from stable-baselines3. The model was developed using 10,000 total simulated games. 

Drake_3_Optuna.zip is a file that contains the trained model. Our agent accesses this file at the start of each game and uses it to determine its next move. This file needs to be present in the current directory for the agent to work. 

Create Submission.py is the file we used to submit our agent to the competition. It contains a copy of the agent1 function in the RL agent.py file. This was written to a file called main.py that could be submitted to the competition. This file also contains a copy of our NStep agent. This was commented out when making main.py so we would not make a mistake when submitting our agent.  

The Folders called Deliverable 1 Code and Deliverable 2 Code have all of our files from the first two parts of the project.  

The zip file called Deliverable3 Submission is what we submitted to the competition. It contains a copy of main.py and Drake_3_Optuna.zip. 