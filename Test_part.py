import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tic_tac_toe
from tic_tac_toe import State, State_test, AI_agent, TD, visualize

'''
Test1 title: State class, _init_ function:
Test design: Directly call the function test = State(), and print its public data
            test.Board_Row, test.Board_Col, test.data, test.winner,
Output must: 12,12, a 12*12 zero matrix, 0
'''
test = State()
print(test.Board_Row,
      test.Board_Col,
      test.data,
      test.winner)


'''
Test2 title: State class, rotate function:
Test design: Let the up 6*4 corner storage 1, the down 6*4 corner storage -1.
            print original 12*12 matrix, then print rotated 12*12 matrix from 1 times to 3 times.
Output must: the 12*12 matrix rotate anti-clockwise from 0 times to 3 times
'''
test = State()
test.data[0:6,4:7] = 1 #up-left corner
test.data[6:12,4:7] = -1 #down-right corner
for n in range(4): #rotate times is from 0 to 3
    print(test.data) #print the matrix
    print('\n')
    test.data = test.rotate(test.data) #rotate the board
    
    
'''
Test3 title: State class, rotate_action function:
Test design: Input the action [4,7], and n is rotation times set to be 0-4.
            print the rotated action.
Output must: [4,7], [7,7], [7,4], [4,4], [4,7]
'''
test = State()
for n in range(5): #rotate times set to be 0-4
    action = test.rotate_action([4,7], n)
    print(action)

    
'''
Test4 title: State class, hash function:
Test design: Set the board as
            |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |10 |11 |
            -----------------------------------------------------
            | 0 ||||||||||||||||| 0 | x | x | x |||||||||||||||||
            -----------------------------------------------------
            | 1 ||||||||||||||||| x | 0 | 0 | x |||||||||||||||||
            -----------------------------------------------------
            | 2 ||||||||||||||||| 0 | 0 | x | 0 |||||||||||||||||
            -----------------------------------------------------
            | 3 ||||||||||||||||| 0 | 0 | x | 0 |||||||||||||||||
            -----------------------------------------------------
            | 4 | x | x | 0 | 0 | x | x | 0 | 0 | x | x | x | 0 |
            -----------------------------------------------------
            | 5 | 0 | 0 | x | x | 0 | x | 0 | x | 0 | x | 0 | 0 |
            -----------------------------------------------------
            | 6 | x | 0 | x | x | 0 | 0 | x | x | 0 | 0 | x | x |
            -----------------------------------------------------
            | 7 | 0 | x | x | 0 | x | 0 | x | 0 | 0 | 0 | x | 0 |
            -----------------------------------------------------
            | 8 ||||||||||||||||| 0 |   |   | 0 |||||||||||||||||
            -----------------------------------------------------
            | 9 ||||||||||||||||| 0 | 0 | x | x |||||||||||||||||
            -----------------------------------------------------
            |10 ||||||||||||||||| x | x | 0 | x |||||||||||||||||
            -----------------------------------------------------
            |11 ||||||||||||||||| x | 0 | x | x |||||||||||||||||
            print the hash value
Output must: hash value consist of string '00202022220020022200022002222022002212022000220010200200200200220220022220022020',
            rotated times n=3
'''    
test = State()
test.data = np.array([[ 0,  0,  0,  0, -1,  1,  1,  1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0, -1, -1,  1, -1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0, -1, -1,  1, -1,  0,  0,  0,  0],
                      [ 1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1],
                      [-1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1, -1],
                      [ 1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1],
                      [-1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                      [ 0,  0,  0,  0, -1,  0,  0, -1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0, -1, -1,  1,  1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  1,  1, -1,  1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  1, -1,  1,  1,  0,  0,  0,  0]] )
test.hash()
    
    
'''
Test5 title: State class, rotate_action function:
Test design: Use the board in Test4. One can check there is only two empty squares left, and no winner occurred.
            print is_end, winner for this original board.
            Then I place cross in the first empty square, coordinate [8,5]. One can check 4 cross in column.
            print is_end, winner for this board.
            Then I replace nought at same square. One can check 5 nought in diagonal.
            print is_end, winner for this board.
            Then I replace the nought to cross at coordinate [6,3], so that break 5 nought in diagonal. 
            And place cross at last square, coordinate [8,6]. One can check game is drawn.
            print is_end, winner for this board. 
Output must: FALSE,0; TRUE,-1; TRUE,1; TRUE,0
'''
print(test.is_end(),test.winner)   #original 
test.winner = 0  #reset the winner
test.data[8,5] = -1 
print(test.is_end(),test.winner)  #4 cross in column
test.winner = 0  #reset the winner
test.data[8,5] = 1
print(test.is_end(),test.winner) #5 nought in diagnal
test.winner = 0  #reset the winner
test.data[6,3] = -1
test.data[8,6] = -1
print(test.is_end(),test.winner)  #drawn 
    
    
'''
Test6 title: State class, get_action_space function:
Test design: Use the Board in Test4, print the result of get_action_space 
Output must: A list consist of [8,5] and [8,6]
'''   
test = State()
test.data = np.array([[ 0,  0,  0,  0, -1,  1,  1,  1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0, -1, -1,  1, -1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0, -1, -1,  1, -1,  0,  0,  0,  0],
                      [ 1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1],
                      [-1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1, -1],
                      [ 1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1],
                      [-1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                      [ 0,  0,  0,  0, -1,  0,  0, -1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0, -1, -1,  1,  1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  1,  1, -1,  1,  0,  0,  0,  0],
                      [ 0,  0,  0,  0,  1, -1,  1,  1,  0,  0,  0,  0]] )
test.get_action_space()   
    
    
'''
Test7 title: State class, next_state function:
Test design: Set the board as
            |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |10 |11 |
            -----------------------------------------------------
            | 0 |||||||||||||||||   |   |   |   |||||||||||||||||
            -----------------------------------------------------
            | 1 |||||||||||||||||   |   |   |   |||||||||||||||||
            -----------------------------------------------------
            | 2 |||||||||||||||||   |   |   |   |||||||||||||||||
            -----------------------------------------------------
            | 3 |||||||||||||||||   |   |   |   |||||||||||||||||
            -----------------------------------------------------
            | 4 |   |   |   |   |   |   |   |   |   |   |   |   |
            -----------------------------------------------------
            | 5 |   | X | O | X |   |   |   |   |   |   |   |   |
            -----------------------------------------------------
            | 6 |   | O |   | O |   |   |   |   |   |   |   |   |
            -----------------------------------------------------
            | 7 |   | X | O | X |   |   |   |   |   |   |   |   |
            -----------------------------------------------------
            | 8 |||||||||||||||||   |   |   |   |||||||||||||||||
            -----------------------------------------------------
            | 9 |||||||||||||||||   |   |   |   |||||||||||||||||
            -----------------------------------------------------
            |10 |||||||||||||||||   |   |   |   |||||||||||||||||
            -----------------------------------------------------
            |11 |||||||||||||||||   |   |   |   |||||||||||||||||
            -----------------------------------------------------
            Try action [6,6] 800 times. All the adjacent square are available.
            print the overall result of next_state.hash()
            Try action [6,2] 800 times. All the adjacent square are occupied.
            print the overall result of next_state.hash()
            Try action [0,4] 800 times. The Up and left adjacent square are outside of the board.
            print the overall result of next_state.hash()
            Try action [11,7] 800 times. The Down and Right adjacent square are outside of the board.
            print the overall result of next_state.hash()
Output must: First case should has 9 diferent result, occurred roughly 400, 50 * 8 respectively
            Second case should has two result, occurred roughtly 400 * 2 times respectively
            Third and fourth case should has 5 diferent result, occurred roughly 400, 250, 50 * 3 respectively
'''   
test = State()
test.data = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0.,-1., 1.,-1., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0.,-1., 1.,-1., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
case1 = []#storage case 1 
for n in range(800):
    case1.append(test.next_state([6,6], -1).hash()[0])
pd.Series(case1).value_counts()
case2 = []#storage case 2 
for n in range(800):
    case2.append(test.next_state([6,2], -1).hash()[0])
pd.Series(case2).value_counts()   
case3 = []#storage case 3 
for n in range(800):
    case3.append(test.next_state([0,4], -1).hash()[0])
pd.Series(case3).value_counts()  
case4 = []#storage case 4
for n in range(800):
    case4.append(test.next_state([11,7], -1).hash()[0])
pd.Series(case4).value_counts()      

    
'''
Test8 title: State class, print_state function:
Test design: Randomly generate the 12*12 matrix, all the elements are -1,0,1.
            print the matrix, and the result of print_state function.
Output must: Only inside the board elements are visualized, 1 is nought, -1 is cross.
'''
test = State()
test.data = np.random.randint(-1,2,size=(12,12))
test.data
test.print_state()
    
    
'''
Test9 title: AI_agent class, _init_ function:
Test design: If directly call the function test = AI_agent(), and print its public data
            test.symbol, test.policy, test.Q_value, test.tau, test.epsilon
            If call the function test = AI_agent(-1), and print its public data
            test.symbol, test.policy, test.Q_value, test.tau, test.epsilon
Output must: First case the results should 1,[],[],[],0.1
            Second case the results should -1,[],[],[],0.1
'''    
test=AI_agent()
print(test.symbol, test.policy, test.Q_value, test.tau, test.epsilon)
test=AI_agent(-1)
print(test.symbol, test.policy, test.Q_value, test.tau, test.epsilon)


'''
Test10 title: AI_agent class, policy_update function:
Test design: The input State set to be default state.
            First case: Without any Q value, Initialize the policy. And print the test.policy.
            Second case: With Given policy(default state) = [6,6], without Q value, policy_update. And print the test.policy.
            Third case: With Q value [5,5] = 1, initialize the policy. And print the test.policy
            Fourth case: With Given policy(default state) = [6,6], With Q value [5,5] = 1, policy_update. And print the test.policy.
Output must: First case: [0,4]; Second case: [6,6]; Third case: [5,5]; Forth case: [5,5]
''' 
S = State()
test = AI_agent()
#First case
test.policy_update(S)
print(test.policy)
#Second
test.policy[S.hash()[0]] = [6,6] #With Given policy(default state) = [6,6]
test.policy_update(S)
print(test.policy)
#Third case
test.policy = dict() #rest
test.Q_value[S.hash()[0] + str(5) + str(5)] = 1 #With Q value [5,5] = 1
test.policy_update(S)
print(test.policy)
#Fourth case
test.policy[S.hash()[0]] = [6,6] #With Given policy(default state) = [6,6]
test.policy_update(S)
print(test.policy)


'''
Test10 title: AI_agent class, get_action function:
Test design: The input State set to be default state.
            First case: Without any policy, call the function for 80000 times. And print the overall results.
            Second case: With Given policy(default state) = [6,6], call the function for 300 times. And print the overall results.
Output must: First case there are 80 results. [0,4],[0,4] will occur roughly 72100 times, other occur 100 times.
            Second case there are 80 results. [6,6],[6,6] will occur roughly 72100 times, other occur 100 times.
''' 
S = State()
test = AI_agent()
#First case
case1 = []
for n in range(80000):
    case1.append(test.get_action(S))
pd.Series(case1).value_counts() 
#Second case
test.policy[S.hash()[0]] = [6,6] #With Given policy(default state) = [6,6]
case2 = []
for n in range(80000):
    case2.append(test.get_action(S))
pd.Series(case2).value_counts() 


'''
Test11 title: TD class, delta function:
Test design: Let the hash_val1 be '01', hash_val2 be '02', and set the tau of hash_val1 be 1.
            First case, both Q_value of (S,A) are initially 0, reward is 0, time is 1.
            Second case, both Q_value of (S,A) are initially 0, reward is -1, time is 1
            Third case, both Q_value of (S,A) are initially 0, reward is -1, time is 10
            Fourth case, Q(S,A)_t = 10, Q(S,A)_(t+1) = 8, reward is 0, time is 1
            Fifth case, Q(S,A)_t = 10, Q(S,A)_(t+1) = 8, reward is 1, time is 1
Output must: Compute the formula by hand, the results should be 0,-1,-0.997,-2.8,-1.8 respectively
''' 
test = TD()
test.player1.tau['01'] = 1 #set the tau of hash_val1 be 1
case1 = test.delta(test.player1, '01', '02', 0, 1) #case1
case2 = test.delta(test.player1, '01', '02', -1, 1) #case2
case3 = test.delta(test.player1, '01', '02', -1, 10) #case3
test.player1.Q_value['01'] = 10 #Q(S,A)_t = 10
test.player1.Q_value['02'] = 8 # Q(S,A)_(t+1) = 8
case4 = test.delta(test.player1, '01', '02', 0, 1) #case 4
case5 = test.delta(test.player1, '01', '02', 1, 1) #case 5
print(case1,case2,case3,case4,case5)


'''
Test12 title: TD class, backup function:
Test design: Let the hash_val be '01'.
            First case, Q_value is initially 0, delta is 1, E is 0
            Second case, Q_value is initially 0, delta is 1, E is 1
            Third case, Q_value is 10, delta is 1, E is 0
            Fourth case, Q_value is 10, delta is 1, E is 0.5
Output must: Compute the formula by hand, the Q_value after backup should be 0,0.2,10,10.1 respectively
''' 
test = TD()
test.backup(test.player1, '01', 1, 0)#case1
case1 = test.player1.Q_value['01']
test.player1.Q_value['01'] = 0 #reset the Q_value be initially 0
test.backup(test.player1, '01', 1, 1)#case 2
case2 = test.player1.Q_value['01']
test.player1.Q_value['01'] = 10 #set Q_value is 10
test.backup(test.player1, '01', 1, 0)#case 3
case3 = test.player1.Q_value['01']
test.player1.Q_value['01'] = 10 #set Q_value is 10
test.backup(test.player1, '01', 1, 0.5)#case 4
case4 = test.player1.Q_value['01']
print(case1,case2,case3,case4)


'''
Test13 title: TD class, episode function:
Test design: Call the function directly.
Output must: The length of 1,2nd list should be the same, 1 element less than 3rd list.
            The length of 4,5th list should be the same, 1 element less than 6th list.
            The last element of 3,6th list should be 'End'.
            The 7th result should be integer -1/0/1.
'''
test = TD()
r1,r2,r3,r4,r5,r6,r7 = test.episode()
print(len(r1),len(r2),len(r3),len(r4),len(r5),len(r6),r7)
print(r3[len(r3)-1],r6[len(r6)-1])


'''
Test14 title: TD class, Q_learning function:
Test design: Test its function using State_test class, which is a 4*4 tic-tac-toe.
            And player wins the game when 3 in row/column/diagonal.
            No rejection for players' choice. 
Output must: First player can always win the game by 3 steps. So the value of initial state S is
                v(S) = q(S,pi(S)) = 0.9*0.9*1 = 0.81
'''
test = TD() #Specially, the episode is generated using State_test class instead of State class
S = State_test() #initial State
h = S.hash()[0] #get the initial State hash value
reward1 = [] #storage the v(S) for player1
for i in range(100000):
    test.Q_learning(i)
    pi = test.player1.policy[h] #pi(S)
    reward1.append(test.player1.Q_value.get(h + str(pi[0]) + str(pi[1]),0))
plt.plot(reward1) 
plt.axhline(0.81,color='black',linestyle='--')
plt.xlabel('episode')
plt.ylabel('v(S)')

