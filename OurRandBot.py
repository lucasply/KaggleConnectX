def act(observation, configuration):
    board = observation.board
    columns = configuration.columns
    return [c for c in range(columns) if board[c] == 0][0]
