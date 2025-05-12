import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tic_tac_toe
from tic_tac_toe import State, State_test, AI_agent, TD


Task = TD() 
S = State() #initial State
h = S.hash()[0] #get the initial State hash value
reward1 = [] #storage the v(S) for player1
reward2 = [] #storage the v(S) for player2
for i in range(1000000,1200000):
    Task.Q_learning(i)
    pi = Task.player1.policy[h] #pi(S)
    reward1.append(Task.player1.Q_value.get(h + str(pi[0]) + str(pi[1]),0))
### View the First player reward
plt.plot(reward1) 
plt.xlabel('episode')
plt.ylabel('v(S)')


