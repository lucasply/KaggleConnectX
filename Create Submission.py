#My_Agent


def my_agent(obs, config):
    from kaggle_environments import evaluate, make, utils
    import numpy as np
    import random

    #Helper Functions for N-Step lookahead

    # Helper function for score_move: gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)
        
    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    # Helper function for minimax: checks if agent or opponent has four in a row in the window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Helper function for minimax: checks if game has ended
    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False

    # Minimax implementation
    def minimax(node, depth, maximizingPlayer, mark, config):
        is_terminal = is_terminal_node(node, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if depth == 0 or is_terminal:
            return NS_Get_heuristic(node, mark, config)
        if maximizingPlayer: #max's turn
            value = -np.inf
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                value = max(value, minimax(child, depth-1, False, mark, config))
            return value
        else: #min's turn
            value = np.inf
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1, config)
                value = min(value, minimax(child, depth-1, True, mark, config))
            return value

    #Get Scores for each possible move using minimax
    def NS_score_move(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        score = minimax(next_grid, nsteps-1, False, mark, config)
        return score

    #This is the heuristic used for N-Step
    def NS_Get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        avg_mid = getAvgMiddle(grid, mark)
        score = 1e1*num_threes - 1e3*num_threes_opp - 1e5*num_fours_opp + 1e7*num_fours + avg_mid
        return score

    def getAvgMiddle(grid, mark):
        yPos = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if(grid[i][j] == mark):
                    yPos.append(i)
        sum = 0
        for i in range(len(yPos)):
            sum += yPos[i]
        avg = 1/((sum / len(yPos) - 3) +.1)
        return avg

    N_STEPS = 3 #Change this value to change how many steps ahead the agent looks
    #Agent for N-Step lookahead
    #This will also keep track of how lng it takes to make a move
    
        # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
        # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [NS_score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
        # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
        # Select at random from the maximizing columns
    if (3 in max_cols):
        return 3
    return random.choice(max_cols)

from kaggle_environments import make, utils
import sys
import inspect
import os
def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)
write_agent_to_file(my_agent, "main.py")

out = sys.stdout
submission = utils.read_file("main.py")
agent = utils.get(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])

print("\nSuccess!" if env.state[0].status == env.state[1].status == "DONE" else "\nFailed...")