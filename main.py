from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch
import numpy as np
import random
import matplotlib.pyplot as plt


#random seed
random_seed = 0
torch.manual_seed(random_seed)  # torch
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random

#hyperparameters
buffer = 80
BATCH_SIZE = 40
rendering = False
epochs = 1000
end_step = 100
gamma = 0.9 #since it may take several moves to goal, making gamma high

#model
model = Q_learning(64, [150,150], 4, hidden_unit)
optimizer = optim.RMSprop(model.parameters(), lr = 1e-2)
# optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0)
criterion = torch.nn.MSELoss()
memory = ReplayMemory(buffer)

model_2 = Q_learning(64, [150,150], 4, hidden_unit)
optimizer_2 = optim.RMSprop(model_2.parameters(), lr = 1e-2)
# optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0)
criterion_2 = torch.nn.MSELoss()
memory_2 = ReplayMemory(buffer)

def trainAlgo(init, marginal_rate_train):
    epsilon = 1

    for i in range(epochs):
        if init==0:
            state = initGrid()
        elif init==1:
            state = initGridPlayer()
        elif init==2:
            state = initGridRand()
        
        status = 1
        step = 0

        reward_memory_1 = [] #goal에 도달했을 때 reward의 한계효용을 주기 위함
        action_memory_1 = []

        reward_memory_2 = []
        action_memory_2 = []
        
        #while game still in progress
        while(status == 1):   
            v_state = Variable(torch.from_numpy(state)).view(1,-1)
            qval = model(v_state)
            qval_2 = model_2(v_state)
            if (np.random.random() < epsilon): #choose random action
                action = np.random.randint(0,4)
                action_2 = np.random.randint(0,4)
            else: #choose best action from Q(s,a) values
                action = np.argmax(qval.data)
                action_2 = np.argmax(qval_2.data)
            #Take action, observe new state S'
            
            action_memory_1.append(action)

            if check_overlap(action_memory_1): #같은 action을 4번 이상할 경우 다른 action을 선택하도록 함
                choices = [i for i in range(4) if i != action]
                action = np.random.choice(choices)

            player = True
            new_state = makeMove(state, action, player)
            v_new_state = Variable(torch.from_numpy(new_state)).view(1,-1)
            reward, reward_obj = getReward(new_state, reward_memory_1, marginal_rate_train)
            reward_memory_1.append(reward_obj)
            memory.push(v_state.data, action, v_new_state.data, reward)

            if (len(memory) < buffer): #if buffer not filled, add to it
                state = new_state
                continue
            
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
            new_state_batch = Variable(torch.cat(batch.new_state))
            reward_batch = Variable(torch.FloatTensor(batch.reward))
            non_final_mask = (reward_batch == -1)
            #Let's run our Q function on S to get Q values for all possible actions
            qval_batch = model(state_batch)
            # we only grad descent on the qval[action], leaving qval[not action] unchanged
            state_action_values = qval_batch.gather(1, action_batch)
            #Get max_Q(S',a)
            with torch.no_grad():
                newQ = model(new_state_batch)
            maxQ = newQ.max(1)[0]

            y = reward_batch
            y[non_final_mask] += gamma * maxQ[non_final_mask]
            y = y.view(-1,1)
            print("Game #: %s" % (i,), end='\r')
            loss = criterion(state_action_values, y)
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.grad.data.clamp_(-1, 1)
            optimizer.step()

            action_memory_2.append(action_2)

            if check_overlap(action_memory_2): #같은 action을 4번 이상할 경우 다른 action을 선택하도록 함
                choices = [i for i in range(4) if i != action_2]
                action_2 = np.random.choice(choices)

            player = False
            new_state = makeMove(new_state, action_2, player)
            v_new_state = Variable(torch.from_numpy(new_state)).view(1,-1)
            #Observe reward
            reward_2, reward_obj_2 = getReward(new_state, reward_memory_2, marginal_rate_train)
            reward_memory_2.append(reward_obj_2)
            memory_2.push(v_state.data, action_2, v_new_state.data, reward_2)
            step +=1

            if (len(memory_2) < buffer): #if buffer not filled, add to it
                state = new_state
                continue

            transitions_2 = memory_2.sample(BATCH_SIZE)
            batch_2 = Transition(*zip(*transitions_2))
            state_batch_2 = Variable(torch.cat(batch_2.state))
            action_batch_2 = Variable(torch.LongTensor(batch_2.action)).view(-1,1)
            new_state_batch_2 = Variable(torch.cat(batch_2.new_state))
            reward_batch_2 = Variable(torch.FloatTensor(batch_2.reward))
            non_final_mask_2 = (reward_batch_2 == -1)
            #Let's run our Q function on S to get Q values for all possible actions
            qval_batch_2 = model_2(state_batch_2)
            # we only grad descent on the qval[action], leaving qval[not action] unchanged
            state_action_values_2 = qval_batch_2.gather(1, action_batch_2)
            #Get max_Q(S',a)
            with torch.no_grad():
                newQ_2 = model_2(new_state_batch_2)
            maxQ_2 = newQ_2.max(1)[0]

            y_2 = reward_batch_2
            y_2[non_final_mask_2] += gamma * maxQ_2[non_final_mask_2]
            y_2 = y_2.view(-1,1)
            print("Game #: %s" % (i,), end='\r')
            loss_2 = criterion_2(state_action_values_2, y_2)
            
            # Optimize the model
            optimizer_2.zero_grad()
            loss_2.backward()
            for p in model_2.parameters():
                p.grad.data.clamp_(-1, 1)
            optimizer_2.step()
            state = new_state

            if reward_2 != -1:
                status = 0
            if step > end_step:
                break
        if epsilon > 0.1:
            epsilon -= (1/epochs)

## Here is the test of AI
def testAlgo(init):
    step = 0
    status = 1
    reward_sum_1 = 0
    reward_memory = []
    reward_sum_2 = 0
    reward_memory_2 = []
    action_memory = []
    action_memory_2 = []

    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()

    if rendering:
        print("Initial State:")
        print(dispGrid(state))

    #while game still in progress
    while(status == 1):
        v_state = Variable(torch.from_numpy(state))
        qval = model(v_state.view(64))
        action = np.argmax(qval.data) #take action with highest Q-value
       
        action_memory.append(action)   
        if check_overlap(action_memory):
            choices = [i for i in range(4) if i != action]
            action = np.random.choice(choices)

        player = True
        state = makeMove(state, action, player)
        reward, reward_obj = getReward(state, reward_memory, marginal_rate_test)
        if reward != -1: #reward와 효용을 따로 계산하기 위함
            reward_sum_1 += reward
        reward_memory.append(reward_obj)

        if rendering:
            print("P1 qval: ", qval)
            print('P1 Move #: %s; Taking action: %s' % (step, action)) 
            print(dispGrid(state))

        v_state = Variable(torch.from_numpy(state))
        qval_2 = model_2(v_state.view(64))
        action_2 = np.argmax(qval_2.data) #take action with highest Q-value

        action_memory_2.append(action_2)
        if check_overlap(action_memory_2):
            choices = [i for i in range(4) if i != action_2]
            action_2 = np.random.choice(choices)

        player = False
        state = makeMove(state, action_2, player)
        reward_2, reward_obj_2 = getReward(state, reward_memory_2, marginal_rate_test)
        if reward != -1:
            reward_sum_2 += reward_2
        reward_memory_2.append(reward_obj_2)

        if rendering:
            print("P2 qval: ", qval_2)
            print('P2 Move #: %s; Taking action: %s' % (step, action_2))
            print(dispGrid(state))

        step += 1 #If we're taking more than end_count actions, just stop, we probably can't win this game
        P1_pair = (reward_memory.count('goal'), reward_memory.count('goal_2'))
        P2_pair = (reward_memory_2.count('goal'), reward_memory_2.count('goal_2'))
        # reward_number_sum = (reward_memory.count('goal')+ reward_memory.count('goal_2'))
        # reward_number_sum_2 = (reward_memory_2.count('goal')+ reward_memory_2.count('goal_2'))
        
        if findLoc(state, np.array([0,1,0,0])) == [] and findLoc(state, np.array([1,0,0,0])) == []:
            if rendering:
                print("P1 reward: ", reward_sum_1)
                print("P2 reward: ", reward_sum_2)
                print("P1 R pair: ", P1_pair)
                print("P2 R pair: ", P2_pair)
            return reward_sum_1, reward_sum_2, P1_pair, P2_pair, step
    
        if (step > end_step):
            # print("Game lost; too many moves.")
            return None, None, None, None, step
            return reward_sum_1, reward_sum_2, P1_pair, P2_pair, step

    print("Reward: %s" % (reward,))

if __name__ == "__main__":
    # game mode
    init = 0
    episode = 100
    marginal_rate_train = 1
    marginal_rate_test = 0.8
    Total_Utility_list = []
    Std_list = []
    episode_list = []
    step_list = []
    print("train epochs: ", epochs)
    print("test episode: ", episode)
    print("gamma: ", gamma)
    print()

    for i in range(10):
        #Train
        marginal_rate_train = round(marginal_rate_train, 1)
        trainAlgo(init, marginal_rate_train)

        #Memory
        P1_reward_memory = []
        P2_reward_memory = []
        P1_pair_memory = []
        P2_pair_memory = []
        step_memory = []
        episode_memory = 0        

        for i in range(episode):
            #Test
            P1_reward, P2_reward, P1_pair, P2_pair, step = testAlgo(init)
            if P1_reward != None and P2_reward != None and P1_pair != None and P2_pair != None:
                P1_reward_memory.append(P1_reward)
                P2_reward_memory.append(P2_reward)
                P1_pair_memory.append(P1_pair)
                P2_pair_memory.append(P2_pair)
                if step <= end_step:
                    episode_memory += 1
                    step_memory.append(step)

        Total_Utility = (np.mean(P1_reward_memory)+np.mean(P2_reward_memory))
        Total_pair = [item for tpl in (P1_pair_memory + P2_pair_memory) for item in tpl]
        SD = np.std(Total_pair)
        Total_Utility_list.append(Total_Utility)
        Std_list.append(SD)
        episode_list.append(episode_memory)
        step_list.append(np.mean(step_memory))

        print("Marginal rate: ", marginal_rate_train)
        # print("P1 ratio: ", P1_pair_memory)
        # print("P2 ratio: ", P2_pair_memory)
        # print("P1 reward: ", P1_reward_memory) 
        # print("P2 reward: ", P2_reward_memory)
        # print("P1 reward mean: ", np.mean(P1_reward_memory))
        # print("P2 reward mean: ", np.mean(P2_reward_memory))
        
        print("Total Utility: ", Total_Utility)
        print("Std: ", SD)
        print("Success Episode: ", episode_memory)
        print("step mean: ", np.mean(step_memory))
        print()

        marginal_rate_train -= 0.1
        plt.plot(P1_reward_memory, label='P1')
        plt.plot(P2_reward_memory, label='P2')
    
    print("Total Utility mean: ", np.mean(Total_Utility_list))
    print("Std mean: ", np.mean(Std_list)) 
    print("Success Episode mean: ", np.mean(episode_list))
    print("Step mean: ", np.mean(step_list))
    print()
