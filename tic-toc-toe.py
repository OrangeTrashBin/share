# Define the game board as a numpy array
board = np.array([['-' for i in range(3)] for j in range(3)])

# Define the state space
state_space = [(board[0,0], board[0,1], board[0,2],
                board[1,0], board[1,1], board[1,2],
                board[2,0], board[2,1], board[2,2])]

# Define the action space
action_space = [(i, j) for i in range(3) for j in range(3) if board[i,j] == '-']

# Define the reward function
def reward_function(state, player, opponent):
    for i in range(3):
        if state[i*3] == state[i*3+1] == state[i*3+2] == player:
            return 10
        if state[i] == state[i+3] == state[i+6] == player:
            return 10
    if state[0] == state[4] == state[8] == player or \
       state[2] == state[4] == state[6] == player:
        return 10
    if '-' not in state:
        return 0
    return 1

# Define the Q-table
Q = dict()
for state in state_space:
    for action in action_space:
        Q[(state, action)] = 0

# Define the Q-learning algorithm
def q_learning(state, player, opponent, alpha, gamma, epsilon):
    if state not in state_space:
        return
    if random.uniform(0,1) < epsilon:
        action = random.choice(action_space)
    else:
        max_value = -1000000
        for a in action_space:
            if Q[(state, a)] > max_value:
                max_value = Q[(state, a)]
                action = a
    row, col = action
    board[row,col] = player
    next_state = (board[0,0], board[0,1], board[0,2],
                  board[1,0], board[1,1], board[1,2],
                  board[2,0], board[2,1], board[2,2])
    reward = reward_function(next_state, player, opponent)
    max_future_reward = -1000000
    for a in action_space:
        if Q[(next_state, a)] > max_future_reward:
            max_future_reward = Q[(next_state, a)]
    Q[(state, action)] = (1-alpha) * Q[(state, action)] + \
                         alpha * (reward + gamma * max_future_reward)
    if reward >= 10 or reward <= -10 or '-' not in next_state:
        return
    q_learning(next_state, opponent, player, alpha, gamma, epsilon)
