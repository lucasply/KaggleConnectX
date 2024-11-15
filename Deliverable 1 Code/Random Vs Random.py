#Random Vs. Random
from kaggle_environments import make, evaluate

env = make("connectx", debug=True)
env.render()
# Two random agents play one game round
env.run(["random", "random"])

# Show the game
env.render(mode="ipython", width=500, height=450)
