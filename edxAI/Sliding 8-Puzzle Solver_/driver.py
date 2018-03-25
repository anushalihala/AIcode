#INPUT: TYPE OF SEARCH, SLIDING 8 PUZZLE BOARD, e.g. >>python driver.py bfs 1,2,5,3,4,0,6,7,8
#OUTPUT: output.txt contains;
# # path_to_goal: the sequence of moves taken to reach the goal (by empty position / element 0)
# # cost_of_path: the number of moves taken to reach the goal
# # nodes_expanded: the number of nodes that have been expanded
# # search_depth: the depth within the search tree when the goal node is found
# # max_search_depth:  the maximum depth of the search tree in the lifetime of the algorithm
# # running_time: the total running time of the search instance, reported in seconds
# # max_ram_usage: the maximum RAM usage in the lifetime of the process as measured by the ru_maxrss attribute in the resource module, 
#PROGRAM IMPLEMENTS DFS, BFS AND A* SEARCH TO FIND SOLTUION

#EXAMPLE
#INPUT: dfs 1,2,5,3,4,0,6,7,8
#OUTPUT:
#path_to_goal: ['Up', 'Left', 'Left']
#cost_of_path: 3
#nodes_expanded: 181435
#search_depth: 3
#max_search_depth: 66211
#running_time: 3.76600003242
#max_ram_usage: None
#(as seen in output file)

#SOME TEST CASES:
#1,2,5,3,4,0,6,7,8
#6,1,8,4,0,2,7,3,5
#8,6,4,2,1,3,5,7,0

import sys
from math import sqrt
from collections import deque
import heapq as hq
import time

class Queue:
    def __init__(self):
        self.order = deque([])  #stores order of elements in Queue (hash values of elements stored in FIFO deque)
        self.storage=dict()     #stores actual elements in Queue (elements stored in dict using their hashval) [improves performance of element search]
    
    def isEmpty(self): 
        return len(self.order) == 0    

    def enqueue(self,p):
        hashval=hash(p)
        if hashval not in self.storage:
            self.order.appendleft(hashval)  #add to tail of FIFO deque
            self.storage[hashval]=p         #store element in dictionary
            updatemaxdepth(p.getlevel())    #updates max_depth

    def dequeue(self):
        #return self.storage[self.order.pop()]
        return self.storage.pop(self.order.pop(),None)

class Stack:
    def __init__(self):
        self.order = []         #stores order of elements in Stack (hash values of elements stored in LIFO list)
        self.storage=dict()     #stores actual elements in Stack (elements stored in dict using their hashval) [improves performance of element search]

    def isEmpty(self): 
        return len(self.order) == 0

    def push(self,p):
        hashval=hash(p)
        if hashval not in self.storage:
            self.order.append(hashval)   #add to top of LIFO list
            self.storage[hashval]=p      #store element in dictionary
            updatemaxdepth(p.getlevel()) #updates max_depth

    def pop(self):
        return self.storage.pop(self.order.pop(),None)
        
class Priority_Queue:
    def __init__(self):
        self.order = []         #stores order of elements in Priority_Queue (hash values of elements stored in heap in descending order). Heap ordered by i)cost of element ii) element's count)
        self.storage=dict()     #stores actual elements in Priority_Queue (elements stored in dict using their hashval) [improves performance of element search]
        self.counter=0          #number of elements in Priority_Queue
    
    def isEmpty(self): 
        return len(self.storage) == 0    

    def enqueue(self,p):
        hashval=hash(p)
        if hashval not in self.storage:
            cost=manhattanpriority(p.getboard())+p.getlevel()           #COST of element is its Manhattan Priority
            hq.heappush(self.order, (cost, self.counter, hashval) );    #element number is used for ordering (since it is unique) if cost is same
            self.storage[hashval]=p                                     #store element in dictionary
            updatemaxdepth(p.getlevel())                                #updates max_depth
        self.counter=self.counter+1    

    def dequeue(self):
        p,c,hv=hq.heappop(self.order)
        return self.storage.pop(hv,None)        

        
class State:
    #stores a particular layout of the board

    def __init__(self, initstate, parent, change, level):
        #2d list containing board arrangement
        self.mystate=initstate
        #reference to board's parent
        self.parent=parent
        #blank tile's change of position from parent to child -> Up,Down,Left or Right
        self.change=change
        #state's level in search tree
        self.level=level

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other): 
        return self.mystate==other.mystate
        
    def __hash__(self): 
        return hash(self.mystate)
    
    def createchildren(self):

        global num_expanded
        num_expanded +=1
        
        #search for position of 0
        i=self.mystate.find('0')

        children=[]
        
		#indexes of surrounding elements
        up=i-n
        down = i+n 
        left=i-1
        right=i+1
       
        if i not in range(0, n): 																   #if empty position (element 0) not in first row
            children.append(  State( swap(self.mystate, up, i), self, "Up", self.level+1 )  ) 	   #Move Up
        if i not in range(boardlen-n, boardlen): 												   #if empty position (element 0) not in last row
            children.append(  State( swap(self.mystate, i, down), self, "Down", self.level+1 )  )  #Move Down
        if i%n!=0:																				   #if empty position (element 0) not in first column
            children.append(  State( swap(self.mystate,left,i), self, "Left", self.level+1 )  )    #Move Left
        if i%n!=(n-1):                                                                             #if empty position (element 0) not in last column
            children.append(  State( swap(self.mystate,i,right), self, "Right", self.level+1 )  )  #Move Right
            
        return children
        
    def getboard(self):    
        return self.mystate
    
    def pathtoroot(self):
        path=[]
        curr_state=self
        while curr_state.level!=0:
            path.append(curr_state.change)
            curr_state=curr_state.parent
        path.reverse()    
        return path
    
    def getlevel(self):
        return self.level
       

def swap(oldstr,i,j):
	#swap positions of ith element and jth element in string
    #assumption: i<j
    newstr=oldstr[:i]+oldstr[j]+oldstr[i+1:j]+oldstr[i]+oldstr[j+1:]
    return newstr
    
def updatemaxdepth(lvl):
    global max_depth
    if lvl>max_depth:
            max_depth=lvl
            
def manhattanpriority(board): #HELPER METHOD to calculate Manhattan priority
    #OUTPUT:
    #The sum of the Manhattan distances (sum of the vertical and horizontal distance) from the blocks to their goal positions, plus the number of moves made so far to get to the search node.
    sum=0
    for count in range(1,9):     #goal position of element = element value
        i=board.find(str(count)) #actual position of element
        #char_index/3 gives row of character
        mpr=abs(count/3 - i/3)
        #char_index%3 gives column of character
        mpc=abs(count%3 - i%3)
        sum=sum+mpr+mpc
    return sum    
 
def maxspace():
    if sys.platform == "win32":
		try:
			import psutil
			print("psutil", psutil.Process().memory_info().rss)
			return psutil.Process().memory_info().rss
		except:
			pass
    else:
		try:
			# Note: if you execute Python from cygwin,
			# the sys.platform is "cygwin"
			# the grading system's sys.platform is "linux2"
			import resource
			#print("resource", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		except:
			pass
    
def outputfile(stateobj):
	#output values to file
    ofh=open("output.txt",'w')
    pathlen= str(len(stateobj))
    ofh.write("path_to_goal: " + str(stateobj)+"\n")
    ofh.write("cost_of_path: " + pathlen +"\n")
    ofh.write("nodes_expanded: " + str(num_expanded)+"\n")
    ofh.write("search_depth: " + pathlen+"\n")
    ofh.write("max_search_depth: " + str(max_depth)+"\n")
    ofh.write("running_time: " + str(runtime)+"\n")
    ofh.write("max_ram_usage: " + str(maxramuse) )
    ofh.close()
    
def bfs(init_state):
    #print "This is bfs"   
    global runtime
    global maxramuse
    start_time = time.time()
    
    frontier = Queue()
    frontier.enqueue(init_state)
    
    explored = set() #set is unordered collection of unique elements, enables faster searches
    
    while not frontier.isEmpty():   
        curr_state=frontier.dequeue()
        explored.add(curr_state)
        
        if curr_state.getboard()==goal_board:
		
            runtime=time.time() - start_time   
            maxramuse=maxspace()
			
            return curr_state.pathtoroot()
            
        for child in curr_state.createchildren():
                
            if child not in explored:
                #enqueue checks if child already in queue, if not child is added
                frontier.enqueue(child) #enqueue also updates max_depth
                                        

def dfs(init_state):
    #print "This is dfs" 
    global runtime
    global maxramuse
    start_time = time.time()
    
    frontier = Stack()
    frontier.push(init_state)
    
    explored = set() #set is unordered collection of unique elements, enables faster searches
    
    while not frontier.isEmpty():   
        curr_state=frontier.pop()
        explored.add(curr_state)
        
        if curr_state.getboard()==goal_board:          
            runtime=time.time() - start_time
            maxramuse=maxspace()
            return curr_state.pathtoroot()
            
        for child in reversed(curr_state.createchildren()): 
                
            if child not in explored:
                #stack checks if child already in queue, if not child is added
                frontier.push(child) #stack also updates max_depth
                                     

def astar(init_state):
    #print "This is astar"    
    global runtime
    global maxramuse
    start_time = time.time()
    
    #Note: Elements ordered by Manhattan Priority: The sum of the Manhattan distances (calculated by manhattanpriority function), and the number of moves made so far to get to the search node (=current level). 
    frontier = Priority_Queue()
    frontier.enqueue(init_state)
    
    explored = set() #set is unordered collection of unique elements, enables faster searches
    
    while not frontier.isEmpty():   
        curr_state=frontier.dequeue()
        explored.add(curr_state)
        
        if curr_state.getboard()==goal_board:
		
            runtime=time.time() - start_time
            maxramuse=maxspace()
			
            return curr_state.pathtoroot()
            
        for child in curr_state.createchildren():
                
            if child not in explored:
                #enqueue checks if child already in queue, if not child is added
                frontier.enqueue(child) #enqueue also updates max_depth

                
#global initializations    
goal_board = "012345678"	#board to be found by search
num_expanded=0				#number of nodes expanded
max_depth = -1				#maximum depth of search 
runtime = 0					#runtime of code
maxramuse=0					#maximum ram usage
boardlen=0 					#number of positions in board
n=0							#length of square board

#creating state object for initial state
ilist = str(sys.argv[2]).split(",")  		 #convert string board (second command line argument) into a list
boardlen=len(ilist)         
n = int(sqrt(boardlen))     	 			 #length of square board
strfromlist = "".join(ilist)  				 #convert board from list to string with no delimiters. (Note: board cells only contain single digits)
istate = State(strfromlist, None, None, 0) 	 #Create State object from raw board

#Select method of search using first command line argument
method = sys.argv[1]
if method=="bfs":
    outputfile(bfs(istate))
elif method=="dfs": 
    outputfile(dfs(istate))
elif method=="ast": 
    outputfile(astar(istate))    
   
    

