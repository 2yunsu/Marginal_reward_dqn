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
    state[0,1] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[3,3] = np.array([1,0,0,0])
    return state

#Initialize player in random location, but keep wall, goal and pit stationary
def initGridPlayer():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    state[1,0] = np.array([0,1,0,0])
    #place goal
    state[1,2] = np.array([1,0,0,0])
    state[1,3] = np.array([1,0,0,0])
    
    a = findLoc(state, np.array([0,0,0,1])) #find grid position of player (agent)
    w = findLoc(state, np.array([0,0,1,0])) #find wall
    g = findLoc(state, np.array([1,0,0,0])) #find goal
    p = findLoc(state, np.array([0,1,0,0])) #find pit

    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return initGridPlayer()
    
    return state

#Initialize grid so that goal, pit, wall, player are all randomly placed
def initGridRand():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[randPair(0,4)] = np.array([0,0,1,0])
    #place pit
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


def makeMove(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    state = np.zeros((4,4,4))

    actions = [[-1,0],[1,0],[0,-1],[0,1],[0,0]]
    #e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0][0] + actions[action][0], player_loc[0][1] + actions[action][1])

    if (new_loc != wall[0]):
        if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
            state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0,0,0,1]))
    if (not new_player_loc):
        state[player_loc[0]] = np.array([0,0,0,1])

    #re-place pit
    for i in range(len(pit)):
        if pit[i] != player_loc[0]:
            state[pit[i]][1] = 1
    #re-place wall
    for i in range(len(wall)):
        if wall[i] != player_loc[0]:
            state[wall[i]][2] = 1
    #re-place goal
    for i in range(len(goal)):
        if goal[i] != player_loc[0]:
            state[goal[i]][0] = 1
    return state

def getLoc(state, level): #오브젝트와 겹치는지 확인
    locations = []
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j][level] == 1):
                locations.append((i,j))
    return locations

def getReward(state, reward_memory, action):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    marginal_rate = 0.9
    n_goal = reward_memory.count('goal')
    n_pit = reward_memory.count('pit')
    for i in range(len(pit)):
        if (player_loc[0] == pit[i]):
            reward = -10.0*marginal_rate**n_pit
            return reward, "pit"
    for i in range(len(goal)):
        if (player_loc[0] == goal[i]):
            reward = 10.0*marginal_rate**n_goal
            return reward, "goal"
    if action == 4:
        return 0, "_"
    else:
        return -1, "_"

    
def dispGrid(state):
    grid = np.zeros((4,4), dtype= str)
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = ' '
    
    if wall:
        for i in range(len(wall)):
            grid[tuple(wall[i])] = 'W' #wall
    if goal:
        for i in range(len(goal)):
            grid[tuple(goal[i])] = '+' #goal
    if pit:
        for i in range(len(pit)):
            grid[tuple(pit[i])] = '-' #pit
    if player_loc:
        grid[tuple(player_loc[0])] = 'P' #player

    return grid