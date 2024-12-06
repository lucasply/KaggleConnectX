
model = None
def agent1(obs, config):
    import random
    import numpy as np
    from stable_baselines3 import PPO 
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    global model 
    
    if model == None:
        model = PPO.load("Drake_3_Optuna")
    
    # Use the best model to select a column
    col, _ = model.predict(np.array(obs['board']).reshape(1, 6,7))
    # Check if selected column is valid
    is_valid = (obs['board'][int(col)] == 0)
    # If not valid, select random move. 
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
