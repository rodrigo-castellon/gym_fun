import time
import gym
import numpy as np

env = gym.make('CartPole-v0')
q_table = [] #Where each line of this Q-table is state, action 1, action 2, Q-value 1, Q-value 2
#Learning parameters
learning_rate = 0.01
gamma = 0.99
r_init = 20

t = 0
i_episode = 0
benchmark = 75
num_past = 0

while True:
    if i_episode % 100 == 0:
        print(q_table)
    #if t > benchmark:
    #    num_past += 1
    observation = env.reset()
    #observation = [round(observation[i],2) for i in observation]
    for t in range(2000): #For each timestep t in episode i_episode...
        found = False
        env.render()
        #Search for state in Q-table and choose action based on action there
        for i in range(len(q_table)):
            if np.array_equal(q_table[i][0],observation):
                found = True
                if q_table[i][3] == q_table[i][4]:
                    action = env.action_space.sample()
                else:
                    action = (np.argmax([q_table[i][3],q_table[i][4]]) - 3) #Action is either 0 or 1
                    q_t = q_table[i][action+3] #Q-value of the action at state s_t
                s_t = i
        else: #If not in Q-table, sample random action
            if found == False:
                action = env.action_space.sample()
                q_table.append([observation,0,1,r_init,r_init])
                #print("Added %s to the Q-table" % action)
                s_t = len(q_table)-1
                a_t = q_table[s_t][action+3]
        found = False
        observation, reward, done, info = env.step(action) #Take action and observe reward
        #observation = [round(observation[i],2) for i in observation]
        #print("%s observed with %s reward" % (observation, reward))
        #Find location of new state s_t+1 (s_tt) in Q-table
        for j in range(len(q_table)):
            if np.array_equal(q_table[j][0],observation):
                found = True
                s_tt = j
                #Q-value updated towards landing state's argmax Q-value
                q_table[s_t][action+3] += learning_rate * ((reward + (gamma * max(q_table[s_tt][3],q_table[s_tt][4]))) - q_table[s_t][action+1])
        else: #If this new state has not been visited before
            if found == False:
                q_table[s_t][action+3] += learning_rate * ((reward + (gamma * r_init)) - q_table[s_t][action+1])
        if done:
            t += t
            print("Episode {} finished after {} timesteps".format(i_episode,t))
            i_episode += 1
            break
print("SUCCESSSSSS (only %s though)" % benchmark)
print(q_table)
print(len(q_table))