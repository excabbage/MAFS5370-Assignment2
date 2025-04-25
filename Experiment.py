import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tic_tac_toe
from tic_tac_toe import State, State_test, AI_agent, TD

Task = TD() 
S = State() #initial State
h = S.hash()[0] #get the initial State hash value
reward1 = [] #storage the v(S) for player1
reward2 = [] #storage the v(S) for player2
for i in range(571960,1000000):
    Task.Q_learning(i)
    pi = Task.player1.policy[h] #pi(S)
    reward1.append(Task.player1.Q_value.get(h + str(pi[0]) + str(pi[1]),0))
### View the First player reward
plt.plot(reward1) 
plt.xlabel('episode')
plt.ylabel('v(S)')
### View the Second player reward
X = np.arange(0,12,1)
Y = np.arange(0,12,1)
Z = np.zeros((12,12))
for i in range(12): #the Fist player occupies [i,j]
    for j in range(12):
        if 3 < i < 8 or 3 < j < 8: #inside the board 
            S2 = State()
            S2.data = np.copy(S.data) #copy the board
            S2.data[i,j] = 1 #First player occupies [i,j]
            h2 = S2.hash()[0]
            pi = Task.player2.policy[h2] #pi(S)
            Z[i,j] = Task.player2.Q_value.get(h + str(pi[0]) + str(pi[1]),0)
X, Y = np.meshgrid(X, Y)
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='winter')
plt.show()
