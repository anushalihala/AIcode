#INPUT: Takes a Grid object as parameter 
#OUTPUT: returns a move (up, down, left or right).
#Program chooses move using min-max algorithm.
#Terminal nodes -> Boards with no more moves
#Evaluation function used to calculate utility of a board (instead of terminal values)

from random import randint
from BaseAI import BaseAI
from Displayer import Displayer
from math import log
import time
# import pdb
        
class PlayerAI(BaseAI):
    def __init__(self):
        self.starttime=time.time()
        
        self.critical=0.2 #max time duration for which code is allowed to execute
        self.lax=0.005 #buffer/lax time
        self.allow=self.critical-self.lax #max time duration for which code is actually executed
        
        self.minuti=float('inf')  #initialization value for beta, +ve infinity
        self.maxuti=-float('inf')   #initialization value for alpha, -ve infinity
        
        self.D = 2 #depth of DFS tree  
        self.abort=False #true if allowed time interval over. used to abort further computation
        self.movereach=False #true if leaf node with optimal minimax value found for current iteration of IDS
        
    def reset(self):
        self.starttime=time.time()
        self.abort=False
        self.D = 2 #IDS started with a depth of 2
        self.movereach=False
        
    def showGrid(self,grid):
        #helper method
        #displays grid on console
        show = Displayer() 
        show.display(grid) 
        
    def griddiff(self,lista,listb): #HELPER METHOD for smoothness
        #calculates differences between tiles of different grids
		
		#lista and listb are 2d lists
        diffamt=0
        for i in xrange(len(lista)):
            for j in xrange(len(lista[0])):
                elementdiff=abs(lista[i][j]-listb[i][j])
                elementdiff=elementdiff if elementdiff>0 else 1
                diffamt = diffamt - log(elementdiff,2) #large differences are penalised
        return diffamt    
    
    def smoothness(self,grid): #HEURISTIC 1 (UNUSED)
        #smoothness represent differences between a tile and all its neighbouring tiles
        s=len(grid)-1
        top=grid[:s] 					#extract top 3 rows (rows 0 up to but not including s)
        bottom=grid[1:] 				#extract bottom 3 rows (rows 1 to last row of grid)
        right=[l[1:] for l in grid] 	#extract 3 rightmost columns (for each row, extract elements from 1 to last element of row)
        left=[l[:s] for l in grid] 		#extract 3 leftmost columns (for each row, extract elements from 0 up to but not including element s)
		
        #smoothness  = vertical smoothness across tiles + horizontal smoothness across tiles
        return self.griddiff(top,bottom)+self.griddiff(left,right)    
   
    def getBool(self,val): #HELPER METHOD
        if val<0:
            return 1
        elif val>0:
            return 0
        else:
            return None

    def fixNones(self,lst): #HELPER METHOD
        #None values are set to be equal to neighbouring values
    
        s=len(lst) #size
        if lst.count(None) == s:
            return '0'*s
        
        for i in  xrange(s):
            if lst[i]==None:
                if i!=0:
                    lst[i]=lst[i-1]
                else:
                    #if lst[i] is first element
                    j=i+1
                    while lst[j]==None:
                        j=j+1
                    lst[i]=lst[j]    
        stroflist=''.join(str(i) for i in lst)
        return stroflist
        
    def monoList(self,inlist): #HELPER METHOD
        #calculates monotonicity of a list of tiles
        
        listsize=len(inlist)
        
        #get list differences (increase or decrease)
        lst=map(lambda x,y:x-y,inlist[:listsize-1],inlist[1:])
		
        #get respective boolean values
		#if x-y<0 => x<y (list is increasing) element is 1, 
		#if x-y>0 => x>y (list is decreasing) element is 0
		#if x=y element is None
        lst=map(self.getBool,lst) 
		
		#replaces None values with a 0 or 1
        lststr=self.fixNones(lst) #returns a string
         
        w=1 #weight (controls degree of penalisation)
        
        ##calculation for monotonicity
        #substr_dict gives weights
        substr_dict=[[0,1,1,1],
                     [0,1,1,-1*w],
                     [0,1,-2*w,0],
                     [0,1,-1*w,0],
                     [1,-1*w,0,0],
                     [1,-1*w,0,-1*w],
                     [1,1,-1*w,0],
                     [1,1,1,0]]
        #lststr interpreted as binary integer and used as index in substr_dict to get weights
        indx = int(lststr,2) 
        #monotonicity of list calculated by weighting and summing list 
        pdtlist=map(lambda x,y:x*y,inlist,substr_dict[indx]) 
        tot= sum(pdtlist) 
        return tot       

    def getcolumn(self,matrix, i): #HELPER METHOD
        #returns column i from matrix
        return [row[i] for row in matrix]
    
    def monoRowCol(self,lst): #HELPER METHOD
        #calculates the monotonicity of a row or column in 2048 grid
        #by
        #calculating list monotonicity of the row/column in both directions and selecting lower value
        return min(self.monoList(lst),self.monoList(lst[::-1])) #L[::-1] reverses list
    
    def monotonicity(self,grid): #HEURISTIC 2
        #calculates monotonicity of a 2048 grid (A monotonic row or column is either increasing or decreasing)
        #by summing across all rows and columns
		
        s=len(grid) #size
        tot=0
        for row in grid:
            tot=tot+self.monoRowCol(row) 

        for i in xrange(s):
            column=self.getcolumn(grid,i)
            tot=tot+self.monoRowCol(column)
        return tot
   
    def numAvailableMoves(self,grid):    
        return len(grid.getAvailableMoves())
   
    def numFreeCells(self,grid):    
        return len(grid.getAvailableCells())
    
    def freespacegrade(self,freenum):	#HELPER METHOD for free
        #divides possible number of free cells in the grid into ranges/classes and
        #assigns weight to each class
     	wt=1
        if freenum>12:
            return 1*wt
        elif freenum>8:
            return 1*wt
        elif freenum>4:
            return 0.25*wt
        elif freenum>2:
            return 0.1*wt
        elif freenum>=0:
            return 0*wt
        else:
            #ERROR
            return 0
         
    def maxn(self,matrix,n):	#HELPER METHOD for bordertop
        #returns sum of max 'n' elements in grid
        flatgrid=[j for i in matrix for j in i]
        flatgrid.sort(reverse=True)
        tot=sum(flatgrid[0:n]) 
        return tot  
       
    def failcondition(self,grid): #HELPER METHOD for willfail
        if self.numAvailableMoves(grid)==0:
            return -30
        else:
            return 0     
    
    def getUDLR(self,grid): #HELPER METHOD
        #returns the rows/columns on all 4 sides
        u=grid[0]
        d=grid[-1]
        l=self.getcolumn(grid,0)
        r=self.getcolumn(grid,-1)
        return [u,d,l,r]
    
    def bestline(self,grid): #calculates heuristics 
		#finds edge row/column with maximum monotonicity, and returns the list's sum 
		#(tries to align tiles against one edge in increasing/decreasing order,
		#extra benefits of increasing tile values on this edge)
		
		#finds edge row/column with maximum tile value, and returns the list's monotonicity (list with max tile should be monotonic)
        
        lst = self.getUDLR(grid)
        
        lst_max=[] #stores max item value of u,d,l and r
        lst_mono=[] #stores monotonicity of u,d,l and r
        for line in lst:
            lst_mono.append(self.monoRowCol(line)) 
            lst_max.append(max(line))
            
        max_mono = max(lst_mono) 
        max_mono_idx = lst_mono.index(max_mono)
        tot=sum(lst[max_mono_idx]) #sums values of row/col with maximum monotonicity
        
        max_max = max(lst_max) #maximum tile value
        if lst_max.count(max_max)==1: #hence maximum tile is not in a corner
            max_max_idx = lst_max.index(max_max)
            mono_of_max=lst_mono[max_max_idx] #monotonicity of row/col with maximum tile
            #penalises grid since monotonicity is low if maximum tile not in corner
        else:
            mono_of_max=0 
        
        #returns index of row/col with maximum monotonicity, 
		#row/col with maximum monotonicity itself, 
        #sums of list's values and 
		#monotonicity of row/col with maximum tile
        return (max_mono_idx,lst[max_mono_idx],tot,mono_of_max)

    def snake(self,grid, bestlineT):	#HEURISTIC 3
        #INPUT: tuple from self.bestline(grid) 
        #OUTPUT: value proportional to minimum tile of bestlst and tile adjacent to it 
        #        [if the tile is less than the minimum tile]
        indx,lst,val,_=bestlineT
        
        #index of minimum tile in lst
        if lst[0]<lst[-1]:
            mindx=0
        else:
            mindx=-1
            
        if indx==0:
            nextlst=grid[1] #if lst is first row, nextlst is second row
        elif indx==1:
            nextlst=grid[-2] #if lst is last row, nextlst is second last row
        elif indx==2:
            nextlst=self.getcolumn(grid,1) #if lst is first column, nextlst is second column
        else:
            nextlst=self.getcolumn(grid,-2) #if lst is last column, nextlst is second last column
        
        if lst[mindx]>=nextlst[mindx]: #nextlst[mindx] is adjacent tile to lst[mindx]
            nextval = nextlst[mindx]
        else:
            return 0
		
        if nextval > 0:
         	wt = log(lst[mindx],2)+3 #weight; log base 2 of minimum tile plus 3
        	nextval = nextval*wt
        
        return nextval       
         
    def eval(self,grid): 
		#INPUT: Grid
		#OUTPUT: Utility of grid		
		
        #smooth = self.smoothness(grid.map)																				#HEURISTIC 1 (UNUSED)		 				

        mono =self.monotonicity(grid.map) 																				#HEURISTIC 2
        
        #bestlst: row/col with maximum monotonicity(out of rows/cols bordering grid), 
        #bestL: sum of tile values of bestlst 
        #monoofmax: monotonicity of row/col with maximum tile (out of rows/cols bordering grid)
        (_,bestlst,bestL,monoofmax)=bestT=self.bestline(grid.map)														#HEURISTIC 10 (monoofmax)
        
        bestLlog=log(bestL,2) if bestL>0 else 1																			#HEURISTIC 4
		
        #assumption -> tiles in bestlst are ordered as it has high monotonicity
        #minitem picks up smallest tile
        minitem=min(bestlst[0],bestlst[-1])
        #minitem positively weighted (in proportion to size of tiles in bestlst) to encourage merges with min tile
        minitemval=minitem*bestLlog 																					#HEURISTIC 5
        
        #maximum tile of bestlst
        maxitem=max(bestlst[0],bestlst[-1]) 																			#HEURISTIC 6
        
        #snakes: a value proportional to minimum tile of bestlst and tile adjacent to it 
        #encourages merges with this adjacent tile, which eventually merges with minimum tile
        snakes=self.snake(grid.map,bestT) 																				#HEURISTIC 3 (snake)
        
        free = self.freespacegrade(self.numFreeCells(grid))*bestL 														#HEURISTIC 7
       
        #log base 2 of sum of top 3 tiles in edge rows and columns 
        bordertop=log(self.maxn(self.getUDLR(grid.map),3),2)
        bordertop=bordertop*bestL 																						#HEURISTIC 8
		
		#summing all heuristics with weights
        total=mono + 2.5*free + bordertop + maxitem + minitemval + snakes + monoofmax/2 
        willfail=abs(total)*self.failcondition(grid) 																	#HEURISTIC 9
        
		#adding willfail heuristic to total (as willfail heuristic is dependent on previous total)
        total=total+willfail
        return total #return utility of grid/node
        
    def getPlayerChildren(self,grid):
        #returns moves (up,down,left,right) player can make from current grid
        for x in range(0,4):
            gridCopy = grid.clone()
            if gridCopy.move(x):
                yield (x,gridCopy)
        
    def getCompChildren(self,grid): 
		#returns moves computer can make from current grid
        #children generated by inserting 2 valued or 4 valued tiles in one of the free cells
        for x,y in grid.getAvailableCells():
            gridCopy = grid.clone()
            gridCopy.insertTile((x,y),2)
            yield gridCopy
            gridCopy = grid.clone()
            gridCopy.insertTile((x,y),4)
            yield gridCopy
    
    def timeguard(self,depth):
        #approximations obtained by benchmarking
        looptime=0.0015 #time to complete current execution of for loop
        evaltime=0.004 #duration of eval function
        rettime=0.001 #time taken to return 
        timeelapsed=time.time()-self.starttime #current time - start time of code
        
		#reqtime is estimate of time required to backtrack from current node to root and return a move to function getMove
        reqtime=timeelapsed + evaltime + rettime*(depth+1) + looptime*depth
        if reqtime > self.allow:
            self.abort=True
    
    def min(self,grid,alpha,beta,depth):       
        #Terminal Test - game ends when no more time, cells or set depth reached
        self.timeguard(depth)
        if self.abort==True:		#if no time to travel further down tree
            self.movereach=False	#set flag to indicate that whole tree not explored (i.e. leaf node with optimal minimax value not found)
            return self.eval(grid)
        elif self.numFreeCells(grid)==0 or depth>=self.D: 	#if node is leaf node (no more moves/game over) OR depth of current iteration of IDS reached, 
            return self.eval(grid)							#return node utility
            
        #Initialize
        minUtility=self.minuti
        
        for child in self.getCompChildren(grid):
            
            (_,utility)=self.max(child,alpha,beta,depth+1)
        
            if utility<minUtility:
                minUtility=utility
            
            #pruning
            if minUtility<=alpha:
                break
            
            if minUtility<beta:
                beta=minUtility
            
            #if premature abortion then all nodes not covered
			#set flag to indicate that whole tree not explored (i.e. leaf node with optimal minimax value not found)
            if self.abort==True:
                self.movereach=False
                return minUtility

        return minUtility
               
    def max(self,grid,alpha,beta,depth=0):
        #Terminal Test
        self.timeguard(depth)
        if self.abort==True:				#if no time to travel further down tree
            self.movereach=False			#set flag to indicate that whole tree not explored (i.e. leaf node with optimal minimax value not found)
            return (None,self.eval(grid))
        elif (self.numAvailableMoves(grid)==0) or depth>=self.D: #if node is leaf node (no more moves/game over) OR depth of current iteration of IDS reached, 
            return (None,self.eval(grid))						 #return node utility
        
        #Initialize
        (maxMove,maxUtility)=(None,self.maxuti)
 
        for move,child in self.getPlayerChildren(grid):
       
            utility=self.min(child,alpha,beta,depth+1)
        
            if utility>maxUtility:
                (maxMove,maxUtility)=(move,utility)
            
            #pruning
            if maxUtility>=beta:
                break
            
            if maxUtility>alpha:
                alpha=maxUtility
            
            #if premature abortion then
			#set flag to indicate that whole tree not explored (i.e. leaf node with optimal minimax value not found)
            if self.abort==True:
                self.movereach=False
                return (maxMove,maxUtility)
        
        #Max at depth 0 either returns inside loop (premature abortion)
        #or after loop - in which case all nodes covered and movereach is true
        self.movereach=True
        return (maxMove,maxUtility)

    def getMove(self, grid):
        #initialization
        self.reset()
        playermove=0
        
        #Max-Min tree explored by IDS (Iterative Deepening Search)
        while self.abort==False:
            (newplayermove,_)=self.max(grid,self.maxuti,self.minuti)
            if self.movereach==True:
                playermove=newplayermove
            self.D = self.D+1    

        return playermove       
        