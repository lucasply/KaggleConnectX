#Random Vs. Random
from kaggle_environments import make, evaluate

#Required for connectx submission
def act(observation, configuration):
    board = observation.board
    columns = configuration.columns
    return [c for c in range(columns) if board[c] == 0][0]

env = make("connectx", debug=True)
env.render()

#Our random agent
def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) 
                   if observation.board[c] == 0])
# Two random agents play one game round
env.run([my_agent, "random"])

# Show the game
env.render(mode="ipython", width=500, height=450)