
# This is the random agent submitted to the competition.
def my_agent(obs, config):
    import numpy as np
    import random
    from kaggle_environments import make, evaluate

    env = make("connectx" , debug=True)
    env.render()
    
    # Return which column to drop a checker (action).
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)

