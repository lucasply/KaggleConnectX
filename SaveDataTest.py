#SaveDataTest

from stable_baselines3 import PPO 

model=PPO.load("SelfTrainData")
print(model.n_epochs)
