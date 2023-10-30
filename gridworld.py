import numpy as np
import pdb

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the "depth" dimension of the grid
# def findLoc(state, obj):
#     locations = []
#     for i in range(0,4):
#         for j in range(0,4):
#             if (state[i,j] == obj).all():
#                 locations.append((i,j))
#     return locations

def findLoc(state, obj):
    locations = []
    if (obj == np.array([0,0,0,1])).all():
        n = 3
    if (obj == np.array([0,0,1,0])).all():
        n = 2
    if (obj == np.array([0,1,0,0])).all():
        n = 1
    if (obj == np.array([1,0,0,0])).all():
        n = 0
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j,n] == obj[n]).all():
                locations.append((i, j))
    return locations

#Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((4,4,4))
    #place player
    state[0,0] = np.array([0,0,0,1])
    
    #place player2
    state[3,3] = np.array([0,0,1,0])
    # state[0,1] = np.array([0,0,1,0])
    #place goal_2
    state[0,1] = np.array([0,1,0,0])
    state[0,2] = np.array([0,1,0,0])
    state[1,0] = np.array([0,1,0,0])
    state[1,1] = np.array([0,1,0,0])
    state[1,2] = np.array([0,1,0,0])
    state[1,3] = np.array([0,1,0,0])
    #place goal
    state[2,0] = np.array([1,0,0,0])
    state[2,1] = np.array([1,0,0,0])
    state[2,2] = np.array([1,0,0,0])
    state[2,3] = np.array([1,0,0,0])
    state[3,1] = np.array([1,0,0,0])
    state[3,2] = np.array([1,0,0,0])
    return state

#Initialize player in random location, but keep goal and goal_2 stationary
def initGridPlayer():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place player2
    state[2,2] = np.array([0,0,1,0])
    #place goal_2
    state[1,1] = np.array([0,1,0,0])
    state[1,0] = np.array([0,1,0,0])
    #place goal
    state[1,2] = np.array([1,0,0,0])
    state[1,3] = np.array([1,0,0,0])
    
    a = findLoc(state, np.array([0,0,0,1])) #find grid position of player (agent)
    w = findLoc(state, np.array([0,0,1,0])) #find player2
    g = findLoc(state, np.array([1,0,0,0])) #find goal
    p = findLoc(state, np.array([0,1,0,0])) #find goal_2

    # if (not a or not w or not g or not p):
    #     #print('Invalid grid. Rebuilding..')
    #     return initGridPlayer()
    
    return state

#Initialize grid so that goal, goal_2, player2, player are all randomly placed
def initGridRand():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place player2
    state[randPair(0,4)] = np.array([0,0,1,0])
    #place goal_2
    state[randPair(0,4)] = np.array([0,1,0,0])
    #place goal
    state[randPair(0,4)] = np.array([1,0,0,0])
    
    a = findLoc(state, np.array([0,0,0,1]))
    w = findLoc(state, np.array([0,0,1,0]))
    g = findLoc(state, np.array([1,0,0,0]))
    p = findLoc(state, np.array([0,1,0,0]))
    #If any of the "objects" are superimposed, just call the function again to re-place
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return initGridRand()
    
    return state


def makeMove(state, action, player):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0,0,0,1]))
    player2_loc = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    goal_2 = findLoc(state, np.array([0,1,0,0]))
    state = np.zeros((4,4,4))

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #e.g. up => (player row - 1, player column + 0)
    if player == True:
        new_loc = (player_loc[0][0] + actions[action][0], player_loc[0][1] + actions[action][1])
        old_loc = (player2_loc[0][0], player2_loc[0][1])
    if player == False:
        new_loc = (player2_loc[0][0] + actions[action][0], player2_loc[0][1] + actions[action][1])
        old_loc = (player_loc[0][0], player_loc[0][1])

    # if (new_loc != wall[0]): #벽 제한
    if ((np.array(new_loc) <= (3,3)).all() and
        (np.array(new_loc) >= (0,0)).all() and
        (new_loc != old_loc)
        ):
        if player == True:
            state[new_loc][3] = 1
            state[old_loc][2] = 1
        if player == False:
            state[new_loc][2] = 1
            state[old_loc][3] = 1
    new_player_loc = findLoc(state, np.array([0,0,0,1]))
    new_player2_loc = findLoc(state, np.array([0,0,1,0]))
    if (not new_player_loc) and (not new_player2_loc):
        state[player_loc[0]] = np.array([0,0,0,1])
        state[player2_loc[0]] = np.array([0,0,1,0])

    #re-place goal_2
    for i in range(len(goal_2)):
        if (goal_2[i] != player_loc[0]) and (goal_2[i] != player2_loc[0]):
            state[goal_2[i]][1] = 1

    for i in range(len(goal)):
        if goal[i] != player_loc[0] and (goal[i] != player2_loc[0]):
                state[goal[i]][0] = 1

    return state

def getLoc(state, level): #오브젝트와 겹치는지 확인
    locations = []
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j][level] == 1):
                locations.append((i,j))
    return locations

def getReward(state, reward_memory, marginal_rate):
    player_loc = getLoc(state, 3)
    goal_2 = getLoc(state, 1)
    goal = getLoc(state, 0)
    player2_loc = getLoc(state, 2)
    n_goal = reward_memory.count('goal')
    n_goal_2 = reward_memory.count('goal_2')

    for i in range(len(goal_2)):
        for j in range(len(player_loc)):
            if (player_loc[j] == goal_2[i]) or (player2_loc[j] == goal_2[i]):
                reward = 10.0*(marginal_rate**n_goal_2)
                return reward, "goal_2"
    for i in range(len(goal)):
        for j in range(len(player_loc)):
            if (player_loc[j] == goal[i]) or (player2_loc[j] == goal[i]):
                reward = 10.0*(marginal_rate**n_goal)
                return reward, "goal"
    else:
        return 0, "move"

def dispGrid(state):
    grid = np.zeros((4,4), dtype= str)
    player_loc = findLoc(state, np.array([0,0,0,1]))
    player2_loc = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    goal_2 = findLoc(state, np.array([0,1,0,0]))
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = ' '
    

    if goal:
        for i in range(len(goal)):
            grid[tuple(goal[i])] = '+' #goal
    if goal_2:
        for i in range(len(goal_2)):
            grid[tuple(goal_2[i])] = '-' #goal_2
    if player2_loc:
        for i in range(len(player2_loc)):
            grid[tuple(player2_loc[i])] = '2' #player2
    if player_loc:
        for i in range(len(player_loc)):
            grid[tuple(player_loc[i])] = '1' #player

    return grid

def check_overlap(lst):
    if len(lst) < 5:
        return False
    return all(lst[-1] == x for x in lst[-5:-1])