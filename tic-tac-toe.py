import numpy as np
import math

class State:
    '''
    The class State include all the environment the agent needed:
    Board_Row: The number of row for the board
    Board_Col: The number of column for the board
    data: storage the board
    Reject_action_chance: A probability that agent's action is rejected. And this action is chosen to be one of 8 adjecent position randomly
    winner: storage the result of the game
    function _init_: set the winner be 0, set the board be 12*12 zero matrix.
    function rotate: rotate the board by 90 degree anti-clockwise.
    function rotate_action: When the board is rotated, the action should also be rotated so that it matches same position.
    function hash: Mapping the state into unique value. I design hash value consist of a string and a int.
    function is_end: Judge whether the state is terminal. If so, update the self.winner according the game rule.
    function get_action_space: Return all the empty position of the board in a list
    function next_stae: A function realizes the interaction between agent and boad.
                        Although I call it next_state, the return is the result of agent action, which is actually the state for competitor.
    function print_state: Visualize the board.
    '''
    Board_Row:int = 12
    Board_Col:int = 12
    data:np.array
    Reject_action_chance = 0.5
    winner:int
 
    def __init__(self):
        '''
        the board is represented by an 12 * 12 matrix,
        1 represents first player occupies the position,
        -1 represents second player occupies the position,
        0 represents an empty position or outside the board.
        '''
        self.data = np.zeros((self.Board_Row, self.Board_Col))
        self.winner = 0     

    def rotate(self, data:np.array) -> np.array:
        '''
        Input the original board.
        This function rotate the Board by 90 degree clockwise.
        It is used to reduce the number of state, because some states are the same in the sense of rotation.
        The function return the board after rotation.
        '''
        rotate_data = np.zeros((self.Board_Row, self.Board_Col))
        loop = self.Board_Row * self.Board_Col + 1 #used in the rotated formula
        for i in range(self.Board_Row):
            for j in range(self.Board_Col):
                if 3 < i < 8 or 3 < j < 8: #check whether outside the boad, because I storage it using 12*12 matrix
                    rotate_j = (( self.Board_Col*(j+1+self.Board_Col*i)%loop ) - 1)%self.Board_Col #This is rotation formula, which can be derived by hand. 
                    rotate_i = (( self.Board_Col*(j+1+self.Board_Col*i)%loop ) - rotate_j - 1)/self.Board_Col
                    rotate_data[int(rotate_i)][int(rotate_j)] = data[i][j] #storage the rotated element.
        return rotate_data
    
    def rotate_action(self, action:list[int,int], n:int) -> list[int,int]:
        '''
        Input the action [i,j], rotate it by 90*n degree clockwise.
        Return the action after rotated.
        '''
        loop = self.Board_Row * self.Board_Col + 1 #used in the rotated formula
        i,j = action #initial action
        for times in range(n): #rotate the action for n times
            rotate_j = (( self.Board_Col*(j+1+self.Board_Col*i)%loop ) - 1)%self.Board_Col #This is rotation formula, which can be derived by hand. 
            rotate_i = (( self.Board_Col*(j+1+self.Board_Col*i)%loop ) - rotate_j - 1)/self.Board_Col
            i = rotate_i
            j = rotate_j
        return [int(i),int(j)]
    
    def hash(self) -> tuple[str,int]:
        '''
        Mapping a board to a string by putting (element + 1) in one line row by row. For example:
        Board  # | a | b | c |  =>  (a+1) (b+1) (c+1) (d+1) (e+1) (f+1) (g+1) (h+1) (i+1)
               # | d | e | f |      just putting together as a string
               # | g | h | i |      
        I rotate the board by 90 degree clockwise 0-3 times, and get there string respectively.
        Then I compare these four string to get "minimum" one, and record the number of rotation from original board to this "minimum" board.
        This record of number of rotation can help to recover the state given hash value.(So it is one to one mapping.)
        This rotation can help decreasing the number of hash value.
        The return is the tuple of string and number of rotation.
        ''' 
        board = self.data #The board
        n = 0 #storage the rotation times.
        hash_val = '' #storage the respective string for original board.
        for i in range(self.Board_Row): #mapping the original board to string
            for j in range(self.Board_Col):
                if 3 < i < 8 or 3 < j < 8: #check whether outside the boad, because I storage it using 12*12 matrix
                   hash_val = hash_val + str(int(board[i,j] + 1)) #put element together one by one
        for k in range(1,4) : #rotate the board from 1 tims to 3 times.
            board = self.rotate(board) #rotate the board
            hash_val_tmp = '' #storage the respective string for current board.
            for i in range(self.Board_Row): #mapping to string
                for j in range(self.Board_Col):
                    if 3 < i < 8 or 3 < j < 8:
                        hash_val_tmp = hash_val_tmp + str(int(board[i,j] + 1))
            if hash_val_tmp < hash_val : #using python string compare function, to get the "minimum" one.
                hash_val = hash_val_tmp
                n = k
        return hash_val, n

    def is_end(self) -> bool:
        '''
        check whether a player has won the game, or it's a draw      
        If someone wins, storage it in the self.winner, and return True.
        If nobody wins now, check whether it's a draw by checking whether there is still 0 in Board.
        '''
        results = [] #storage the sum of each row, column. If someone wins, the sum should reach 4 or -4.
        # check row, 4 in a row
        for i in range(self.Board_Row):
            for j in range(self.Board_Col - 4 + 1):
                if 3 < i < 8 or j == 4: #The starting position of 4 in row must satisfies these, so that inside the board
                    results.append(np.sum(self.data[i,j:j+4]))
        # check columns, 4 in a column
        for j in range(self.Board_Col):
            for i in range(self.Board_Row - 4 + 1):
                if i == 4 or 3 < j < 8: #The starting position of 4 in column must satisfies these, so that inside the board
                    results.append(np.sum(self.data[i:i+4,j]))
        results_D = [] #storage the sum of each diagonal. If someone wins, the sum should reach 5 or -5.
        for i in range(self.Board_Row - 5 + 1):
            for j in range(self.Board_Col - 5 + 1):
                if i < 4 and 3 < j < (5+i): #The starting position is restrict in Up 4*4 triangle position so that inside the board
                    results_D.append(self.data[i,j] + self.data[i+1,j+1] + self.data[i+2,j+2] + 
                               self.data[i+3,j+3] + self.data[i+4,j+4]) 
                if 3 < i < (5+j) and j < 4: #The starting position is restrict in Left 4*4 triangle position so that inside the board
                    results_D.append(self.data[i,j] + self.data[i+1,j+1] + self.data[i+2,j+2] + 
                               self.data[i+3,j+3] + self.data[i+4,j+4])             
        for i in range(self.Board_Row - 5 + 1):
            for j in range(4, self.Board_Col):
                if i < 4 and (6-i) < j < 8: #The starting position is restrict in Up 4*4 triangle position so that inside the board
                    results_D.append(self.data[i,j] + self.data[i+1,j-1] + self.data[i+2,j-2] + 
                               self.data[i+3,j-3] + self.data[i+4,j-4]) 
                if 3 < i < (8+8-j) and 7 < j : #The starting position is restrict in Right 4*4 triangle position so that inside the board
                    results_D.append(self.data[i,j] + self.data[i+1,j-1] + self.data[i+2,j-2] + 
                               self.data[i+3,j-3] + self.data[i+4,j-4])                     
        # whether a player has won
        for result in results: # check row, column
            if result == 4 :
                self.winner = 1
                return True
            if result == -4 :
                self.winner = -1
                return True
        for result in results_D: #check diagonal
            if result == 5 :
                self.winner = 1
                return True
            if result == -5 :
                self.winner = -1
                return True            
        # whether it's a draw
        for i in range(self.Board_Row):
            for j in range(self.Board_Col):
                if 3 < i < 8 or 3 < j < 8: # inside the board.
                    if self.data[i,j] == 0 : #There is still empty positon.
                        return False
        return True #There is no empty position. End the game.
        
    def get_action_space(self) -> list[list[int,int]]:
        '''
        Given self.data, the 0 position inside the board are the available action [i,j] for this state.
        Return the list of all available action.
        '''
        action_space = []
        for i in range(self.Board_Row):
            for j in range(self.Board_Col):
                if 3 < i < 8 or 3 < j < 8: #inside the board
                    if self.data[i,j] == 0 :
                        action_space.append([i,j])
        return action_space

    def next_state(self, action:list[int,int], symbol):
        '''
        symbole represents player: 1 or -1
        action is a vector [i, j] represents the position player wants to occupy.
        If Reject_action_change,50%, happen, the action will be randomly chosen from the 8 adjacent square. 
        If position [i, j] is already occupied, return unchanged state.
        else put chessman symbol in position v
        return the State() class.
        '''
        if np.random.uniform(0,1) < self.Reject_action_chance : #with 50% chance, the the player's action is not accepted
            adjacent = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]] #eight square adjacent to the chosen one
            index = round(np.random.uniform(1,len(adjacent))) - 1 #randomly choose adjacent one
            action[0] = action[0] + adjacent[index][0]
            action[1] = action[1] + adjacent[index][1]
        if not 3 < action[0] < 8 or not 0 <= action[1] <= 11 :
            if not 3 < action[1] < 8 or not 0 <= action[0] <= 11:
                return self  #outside the board action, forfeit 
        if self.data[action[0], action[1]] != 0 :     
            return self # Not a empty position, forfeit
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[action[0], action[1]] = symbol
        return new_state

    def print_state(self):
        '''
        Visualize the board
        1 is nought
        -1 is corss
        0 is '|' when outside the board, represents prohibite
        0 is ' ' when inside the board, represents empty.
        And I add the coordinate at the Top and Left.
        '''
        print('|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |10 |11 |')
        for i in range(self.Board_Row):
            print('-----------------------------------------------------')
            if i < 10 :
                out = '| ' + str(i) + ' '
            else :
                out = '|' + str(i) + ' '
            for j in range(self.Board_Col):
                if not 3 < i < 8 and not 3 < j < 8:
                    token = '||||'
                elif self.data[i, j] == -1:
                    token = '| x '
                elif self.data[i, j] ==  1:
                    token = '| 0 '
                else:
                    token = '|   '
                out += token
            out += '|'
            print(out)
        print('-----------------------------------------------------')
        
class State_test:
    '''

    '''
    Board_Row:int = 4
    Board_Col:int = 4
    data:np.array
    winner:int
 
    def __init__(self):
        self.data = np.zeros((self.Board_Row, self.Board_Col))
        self.winner = 0     

    def rotate(self, data:np.array) -> np.array:
        rotate_data = np.zeros((self.Board_Row, self.Board_Col))
        loop = self.Board_Row * self.Board_Col + 1 #used in the rotated formula
        for i in range(self.Board_Row):
            for j in range(self.Board_Col):
                rotate_j = (( self.Board_Col*(j+1+self.Board_Col*i)%loop ) - 1)%self.Board_Col #This is rotation formula, which can be derived by hand. 
                rotate_i = (( self.Board_Col*(j+1+self.Board_Col*i)%loop ) - rotate_j - 1)/self.Board_Col
                rotate_data[int(rotate_i)][int(rotate_j)] = data[i][j] #storage the rotated element.
        return rotate_data
    
    def rotate_action(self, action:list[int,int], n:int) -> list[int,int]:
        loop = self.Board_Row * self.Board_Col + 1 #used in the rotated formula
        i,j = action #initial action
        for times in range(n): #rotate the action for n times
            rotate_j = (( self.Board_Col*(j+1+self.Board_Col*i)%loop ) - 1)%self.Board_Col #This is rotation formula, which can be derived by hand. 
            rotate_i = (( self.Board_Col*(j+1+self.Board_Col*i)%loop ) - rotate_j - 1)/self.Board_Col
            i = rotate_i
            j = rotate_j
        return [int(i),int(j)]
    
    def hash(self) -> tuple[str,int]:
        board = self.data #The board
        n = 0 #storage the rotation times.
        hash_val = '' #storage the respective string for original board.
        for i in range(self.Board_Row): #mapping the original board to string
            for j in range(self.Board_Col):
                 hash_val = hash_val + str(int(board[i,j] + 1)) #put element together one by one
        for k in range(1,4) : #rotate the board from 1 tims to 3 times.
            board = self.rotate(board) #rotate the board
            hash_val_tmp = '' #storage the respective string for current board.
            for i in range(self.Board_Row): #mapping to string
                for j in range(self.Board_Col):
                    hash_val_tmp = hash_val_tmp + str(int(board[i,j] + 1))
            if hash_val_tmp < hash_val : #using python string compare function, to get the "minimum" one.
                hash_val = hash_val_tmp
                n = k
        return hash_val, n

    def is_end(self) -> bool:
        results = [] #storage the sum of each row, column. If someone wins, the sum should reach 3 or -3.
        # check row, 3 in a row
        for i in range(self.Board_Row):
            for j in range(self.Board_Col - 3 + 1):
                    results.append(np.sum(self.data[i,j:j+3]))
        # check columns, 3 in a column
        for j in range(self.Board_Col):
            for i in range(self.Board_Row - 3 + 1):
                results.append(np.sum(self.data[i:i+3,j]))
        #check in diagnol
        for i in range(self.Board_Row - 3 + 1):
            for j in range(self.Board_Col - 3 + 1):
                results.append(self.data[i,j] + self.data[i+1,j+1] + self.data[i+2,j+2])     
        for i in range(self.Board_Row - 3 + 1):
            for j in range(2, self.Board_Col):
                results.append(self.data[i,j] + self.data[i+1,j-1] + self.data[i+2,j-2])                     
        # whether a player has won
        for result in results: # check row, column
            if result == 3 :
                self.winner = 1
                return True
            if result == -3 :
                self.winner = -1
                return True        
        # whether it's a draw
        for i in range(self.Board_Row):
            for j in range(self.Board_Col):
                if self.data[i,j] == 0 : #There is still empty positon.
                        return False
        return True #There is no empty position. End the game.
        
    def get_action_space(self) -> list[list[int,int]]:
        action_space = []
        for i in range(self.Board_Row):
            for j in range(self.Board_Col):
                if self.data[i,j] == 0 :
                        action_space.append([i,j])
        return action_space

    def next_state(self, action:list[int,int], symbol):
        if self.data[action[0], action[1]] != 0 :     
            return self # Not a empty position, forfeit
        new_state = State_test()
        new_state.data = np.copy(self.data)
        new_state.data[action[0], action[1]] = symbol
        return new_state

    def print_state(self):
        print('|   | 0 | 1 | 2 | 3 |')
        for i in range(self.Board_Row):
            print('---------------------')
            out = '| ' + str(i) + ' '
            for j in range(self.Board_Col):
                if self.data[i, j] == -1:
                    token = '| x '
                elif self.data[i, j] ==  1:
                    token = '| 0 '
                else:
                    token = '|   '
                out += token
            out += '|'
            print(out)
        print('---------------------')

class AI_agent:
    '''
    The class AI_agent include all the fitted model:
    symbol: 1 or -1, represants first player or second player respectively
    policy: Storage the deterministic policy. The index is string part of State.hash(). Action storaged in policy is adjusted by rotate same times in State.hash().
    Q_value: Storage the estimate Q value for (S,A). The index is hash value of (S,A).
    tau: Storage the most recent time when visit (S,A). This is used to add bonus reward for the (S,A) which was visited long ago.
    epsilon: A probability that agent takes random action. I use epsilon greedy policy to generate episode. 
    function _init_: Set player's symbol. Set the epsilon.
    function policy_update: Use self.Q_value to update the deterministic greedy action in the self.policy. 
    function get_action: Given state, obtain the policy action and episode action respectively.
    '''
    symbol:int 
    policy = dict() 
    Q_value = dict() #The index is hash value of state-action pair hash value
    tau = dict() #Storage the time when agent visitted state-action pair most recently.
    epsilon:float
    
    def __init__(self, symbol:int=1, epsilon:float=0.1):
        self.symbol = symbol
        self.epsilon = epsilon
        
    def policy_update(self, state:State):
        '''
        Fisrt, the benchmark action is the policy action. If there is no exist policy action, using the first action in the action_space.
        Second, scan through the action_space to get the maximum Q
        Third, update the policy with this action. The action is adjusted by matching it to rotatied state in State.hash(). So that the hash value of (S,A) is unique.
        '''
        action_space = state.get_action_space()
        hash_val,n = state.hash() #getting the number of rotation
        if self.policy.get(hash_val) : #get benchmark. If policy doesn't exist, using first action in action_space.
            action = self.policy[hash_val]
        else :
            action = action_space[0] 
            action = state.rotate_action(action, n) #adjusted the action so that hash value of (S,A) is unique in the sense of rotation        
        Q = self.Q_value.get(hash_val+str(action[0])+str(action[1]), 0) #Q value for (S,A)
        for action_tmp in action_space:
            action_tmp = state.rotate_action(action_tmp, n) #adjusted the action so that hash value of (S,A) is unique in the sense of rotation   
            Q_tmp = self.Q_value.get(hash_val+str(action_tmp[0])+str(action_tmp[1]), 0)
            if Q < Q_tmp: #Get the Maximum Q
                action = action_tmp
                Q = Q_tmp
        self.policy[hash_val] = action #update the greedy policy in the self.policy. There is no return of this function.
    
    def get_action(self, state:State) -> tuple[list[int,int],list[int,int]]:
        '''
        Given state,
        First, obtain policy action. If policy doesn't exist, use the first action in action_space
        Second,, obtain epsilon action by randomly choosing in action_space
        Both actions are adjusted by matching it to rotatied state in State.hash(). So that the hash value of (S,A) is unique.
        Return two action 
        '''
        action_space = state.get_action_space()
        hash_val,n = state.hash() #getting the number of ratation
        if self.policy.get(hash_val) : #get greedy action. If policy doesn't exist, using first action in action_space.
            pi_action = self.policy[hash_val]
        else :
            pi_action = action_space[0] 
            pi_action = state.rotate_action(pi_action, n) #adjusted the action so that hash value of (S,A) is unique in the sense of rotation     
        if np.random.uniform(0,1) < self.epsilon : #If epsilon happen, get random action, else using greedy action
            index = round(np.random.uniform(1,len(action_space))) - 1 #randomly choose action
            mu_action = action_space[index]
            mu_action = state.rotate_action(mu_action, n) #adjusted the action so that hash value of (S,A) is unique in the sense of rotation    
        else :
            mu_action = pi_action
        return pi_action,mu_action

class TD:
    '''
    
    '''
    player1 = AI_agent(1)
    player2 = AI_agent(-1)
    step_size = 0.2
    gamma = 0.9
    lamb = 0.9
    kapa = 0.001

    def delta(self, agent:AI_agent(), hash_val1:str, hash_val2:str, R:int, time:int) -> float:
        '''
        hash_val1 is the hash value of the (S_t,A_t)
        hash_val2 is the hash value of the (S_(t+1),A)
        R is the reward of (S_t,A_t)
        time is current time when call this function. This is used to get the bonus of reward:
            R_bouns = R + kapa * (How long since last visited this (S_t,A_t))
        Calculate the delta by
            ( R_bouns + gamma * Q(S_(t+1),A) ) - Q(S_t,A_t)
        return the delta
        '''
        old_Q = agent.Q_value.get(hash_val1, 0) #Q(S,A)
        new_Q = agent.Q_value.get(hash_val2, 0) #Q(S',A)
        reward = R + self.kapa * np.sqrt(time - agent.tau.get(hash_val1,0))
        delta = reward + self.gamma * new_Q - old_Q
        return delta
    
    def backup(self, agent:AI_agent(), hash_val:str, delta:float, E:float) :
        '''
        hash_val is the hash value of the state-action pair
        delta is the difference of old Q value and target Q value
        E is the eligible trace or state-action pair during the episode
        Backup formula:
                Q = Q + alpha * E * delta
        no return for this function
        '''
        old_Q = agent.Q_value.get(hash_val, 0) #Q(S,A)
        agent.Q_value[hash_val] = old_Q + self.step_size * E * delta #update Q(S,A)
        
    def episode(self) -> tuple[list,list,list,list,list,list,int] :
        '''
        let player1 and player2 play the game until the terminal.
        And storage the states, (S,A) pairs they visit.
        They play the game using mu_policy, and I also stroage there pi_policy for every states.
        return states, pairs player1 visited, and pi_policy of player1;
                states, pairs player2 visited, and pi_policy of player2;
                the winner of game.
        '''
        player1_state = [] #storage the states player1 observed in episode
        player1_pair = []#storage the state-action pairs player1 observed in episode
        player1_Greedy = [] #storage the (S,A) where A=argmax Q(S,a) for player1.
        player2_state = [] #storage the states player2 observed in episode
        player2_pair = []#storage the state-action pairs player2 observed in episode
        player2_Greedy = [] #storage the (S,A) where A=argmax Q(S,a) for player2.
#        S = State()# Starting state
        S = State_test()# Starting state , used for intergral testing       
        while not S.is_end():
            player1_state.append(S) #player1 play first, visit the board
            pi_action,mu_action = self.player1.get_action(S)
            hash_val,n = S.hash() #getting the hash value of board
            player1_pair.append(hash_val+str(mu_action[0])+str(mu_action[1]))            
            player1_Greedy.append(hash_val+str(pi_action[0])+str(pi_action[1]))
            mu_action = S.rotate_action(mu_action, (4-n)%4) #adjusted back the action so that it match the real board.
            S = S.next_state(mu_action, self.player1.symbol) #play1 take action, then player2 play
            if S.is_end() :
                break
            player2_state.append(S) #player2's turn, visit the board
            pi_action,mu_action = self.player2.get_action(S)
            hash_val,n = S.hash() #getting the hash value of board            
            player2_pair.append(hash_val+str(mu_action[0])+str(mu_action[1]))            
            player2_Greedy.append(hash_val+str(pi_action[0])+str(pi_action[1]))
            mu_action = S.rotate_action(mu_action, (4-n)%4) #adjusted back the action so that it match the real board.
            S = S.next_state(mu_action, self.player2.symbol)
        ###The terminal state. Because whatever action agent takes, its Q value should always be 0.
        player1_Greedy.append('End') #This is not the hash value of any state-action pair. I just use it to extend the length by 1.
        player2_Greedy.append('End')
        return player1_state,player1_pair,player1_Greedy,player2_state,player2_pair,player2_Greedy,S.winner
    
    def Q_learning(self, current_times:int) :
        '''
        current_times is how many times agent play the game
        First, generate the episode by self.episode()
        Then use Q(lambda) method to estimate the Q value for every (S,A) agents visited
        Then policy update for both agents.
        Lastly, update the tau for every (S,A) agents visited
        '''
        ### First, generate the episode
        player1_state,player1_pair,player1_Greedy,player2_state,player2_pair,player2_Greedy,winner = self.episode()
        ### initial the eligibility trace
        E_trace1 = np.zeros( len(player1_pair) ) #storage the eligibility trace for state player1 observed
        E_trace2 = np.zeros( len(player2_pair) ) #storage the eligibility trace for state player2 observed
        ### value evaluation
        ###player1 Q value evaluation
        for n in range( len(player1_pair) ) : 
            if( n == len(player1_pair)-1 ): #get the reward of each step
                R = winner * self.player1.symbol 
            else:
                R = 0
            delta = self.delta(self.player1, player1_pair[n], player1_Greedy[n+1], R, current_times) #delta of Q(Sn,An)
            E_trace1[n] = 1 #replacing traces
            for t in range( len(player1_pair) ) : #update all the Q(S,A) using delta
                if(delta * E_trace1[t] !=0): #From the backup formula, only when this non-zero, would the backup meanful
                    self.backup(self.player1, player1_pair[t], delta, E_trace1[t]) #Q lambda method
                if player1_pair[n] == player1_Greedy[n] :# traces update
                    E_trace1[t] = self.lamb * self.gamma * E_trace1[t] #exploitation case
                else :
                    E_trace1[t] = 0#exploration case
        ###player2 Q value evaluation
        for n in range( len(player2_pair) ) :
            if( n == len(player2_pair)-1 ): #get the reward of each step
                R = winner * self.player2.symbol
            else:
                R = 0
            delta = self.delta(self.player2, player2_pair[n], player2_Greedy[n+1], R, current_times) #delta of Q(Sn,An)
            E_trace2[n] = 1 #replacing traces
            for t in range( len(player2_pair) ) : #update all the Q(S,A) using delta
                if(delta * E_trace2[t] !=0): #From the backup formula, only when this non-zero, would the backup meanful
                    self.backup(self.player2, player2_pair[t], delta, E_trace2[t]) #Q lambda method
                if player2_pair[n] == player2_Greedy[n] :# traces update
                    E_trace2[t] = self.lamb * self.gamma * E_trace2[t] #exploitation case
                else :
                    E_trace2[t] = 0#exploration case
        ### policy improvement. I use batch update, because there are no same state in the episode, so it is same for batch update and update immediately
        for n in range( len(player1_state) ) : #player1 policy update
            self.player1.policy_update(player1_state[n])
        for n in range( len(player2_state) ) : #player2 policy update
            self.player2.policy_update(player2_state[n])
        ### update the most recent time when agents visit these state-action pairs
        for n in range( len(player1_pair) ) :
            self.player1.tau[player1_pair[n]] = current_times #memory the most recently time for visited state-action pairs
        for n in range( len(player2_pair) ) :
            self.player2.tau[player2_pair[n]] = current_times #memory the most recently time for visited state-action pairs      

def visualize(AI:AI_agent(), first:bool) :
    S = State_test()# initial board
    if first :
        S.print_state()
        #input (i,j) to put a chessman
        #|   | 0 | 1 | 2 |
        #| 0 |
        #| 1 |
        #| 2 |
        action = input("Input your position: ")
        action = [int(num) for num in action.split(',')]
        S = S.next_state(action, (1 if first else -1))
    while not S.is_end():
        S.print_state()
        hash_val,n = S.hash()
        action = AI.policy[hash_val]
        action = S.rotate_action(action, (4-n)%4)
        S = S.next_state(action, AI.symbol)
        if S.is_end() :
            break    
        S.print_state()
        #input (i,j) to put a chessman
        #|   | 0 | 1 | 2 |
        #| 0 |
        #| 1 |
        #| 2 |
        action = input("Input your position: ")
        action = [int(num) for num in action.split(',')]
        S = S.next_state(action, (1 if first else -1))
    S.print_state()
    
