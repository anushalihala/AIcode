#INPUT: Sudoku board (as a string of numerals, 0 -> blank)
#OUTPUT: Filled Sudoku board
#Program uses ARC CONSISTENCY and BACKTRACKING SEARCH to solve Sudoku as a CSP

#EXAMPLE
#INPUT: 010020300004005060070000008006900070000100002030048000500006040000800106008000000
#OUTPUT: 815627394924385761673491528186952473457163982239748615591236847342879156768514239 BTS 
#(as seen in output file)

#MORE TEST CASES
#010020300002003040050000006004700050000100003070068000300004090000600104006000000
#010020300002003040080000006004700030000600008070098000300004090000800104006000000

import sys
import itertools as it
import pdb
from collections import deque
import heapq as hq

def get_indx(decint):
    #converts linear indices [0-80] to 2d indices [0-8][0-8]
    row = decint/9
    col = decint%9
    return str(row)+str(col)
    
class Sudoku_CSP:
    def __init__(self,sb_list):
        self.s=81 #size
        self.board=dict()      #actual values in Sudoku board
        self.domain=dict()     #Possible value each cell in Sudoku board can take
        self.neighbours=dict() #CONSTRAINT; all neighbours must be different
        
        for i in xrange(self.s):
            #storing values from linear string Sudoku board into custom data structure (dictionary with rowcol as index)
            idx=get_indx(i)                       #idx is a two character string with first value=row and second value=column
            val=int(sb_list[i])
            self.board[idx]=val
            
            if val==0:
                self.domain[idx]=set(range(1,10)) #initial domain -> if cell empty, can take any value from 1 to 9
            else:
                self.domain[idx]=set([val])       #initial domain -> if cell filled, can take only that value
                
            self.neighbours[idx]=self.compute_neighbours(idx)
            
        #TESTING
        # print self.board
        # print self.domain
        # print self.neighbours
        self.revise_all_domain()
        
    def get_r_neighbours(self,idx):
        #get neighbours from same row
        r_list=[]
        for i in xrange(9):
            if i==int(idx[1]): #skipping column of cell
                continue
            r_list.append(idx[0]+str(i)) 
        return r_list
        
    def get_c_neighbours(self,idx):
        #get neighbours from same column
        c_list=[]
        for i in xrange(9):
            if i==int(idx[0]): #skipping row of cell
                continue
            c_list.append(str(i)+idx[1])
        return c_list
        
    def get_b_neighbours(self,idx):
        r=int(idx[0])
        c=int(idx[1])
        b_list=[]
        
        #getting box's row values
        row_vals=range(0,r%3)+range((r%3)+1,3)       #getting range
        row_vals= map(lambda x: x+3*(r/3), row_vals) #translating
        
        #getting box's column values
        col_vals=range(0,c%3)+range((c%3)+1,3)       #getting range
        col_vals= map(lambda x: x+3*(c/3), col_vals) #translating
        
        #combining lists
        for i in row_vals:
            for j in col_vals:
               b_list.append(str(i)+str(j))
        return b_list

    def compute_neighbours(self,idx):   
        return self.get_r_neighbours(idx)+self.get_c_neighbours(idx)+self.get_b_neighbours(idx)
    
    def get_neighbours(self,idx):
        return self.neighbours[idx]
    
    def revise_idx_domain(self,idx):
        #revises domains of all of idx's neighbours
        val = self.board[idx]
        
        for item in self.neighbours[idx]:
            if val in self.domain[item]:       #domain[pos] returns set of possible values that can occupy pos
                self.update_idx_D(item,val)    #remove item from domain
                # print item,self.domain[item] #TESTING

    def revise_all_domain(self):
        for i in xrange(self.s):
            idx=get_indx(i)
            val=self.board[idx]
            if not val==0:
                self.revise_idx_domain(idx)
    
    def get_arcs(self):
        #getting binary constraints
        lst=[]
        for i in xrange(self.s):
            idx=get_indx(i)
            for item in self.neighbours[idx]:
                arc=(idx,item)
                lst.append(arc)
        return lst
    
    def is_goal(self):
        #all sudoku cells assigned value (no board cell is 0)
            for i in xrange(self.s):
                idx =get_indx(i)
                if self.board[idx]==0:
                    return False
            return True
    
    def check_idx_D(self,idx):
        #return true if domain of idx is empty
        if len(self.domain[idx])==0:
            return True
        else:
            return False
    
    def is_game_over(self,lst=None):
        #return true if any domain in specified cells (i.e. in lst) is empty
        if lst is None:
            lst=[get_indx(i) for i in xrange(self.s)]
              
        for idx in lst:
            if self.check_idx_D(idx):
                return True
        return False
    
    def get_idx_D(self,idx):
        #return domain of idx as list
        return list(self.domain[idx])
        
    def update_idx_D(self,idx,val):
        #update domain of idx by removing val
        self.domain[idx].remove(val)
        
    def print_board(self):
        #return Sudoku board as a linear string
        board_str=""
        for i in xrange(self.s):
            idx=get_indx(i)
            board_str=board_str+str(self.board[idx])
        return board_str
    
    def domain_to_board(self):
        #assigns domain values to board (domain must contain single value for each cell)
        for i in xrange(self.s):
            idx=get_indx(i)
            if self.board[idx]==0:
                
                if len(self.domain[idx])==1:
                    self.board[idx]=self.domain[idx].pop()
    
    def set_idx_board(self,idx,val):
        #set value 'val' at idx in Sudoku board
        self.board[idx]=val
        
        #store previous domains of all neighbours of idx in a list (as a neighbour index, domain pair)
        prev_domains=[]
        for item in self.neighbours[idx]:
            prev_domains.append((item,self.get_idx_D(item)))
        
        self.revise_idx_domain(idx) #forward checking
        
        return prev_domains
    
    def del_idx_board(self,idx,prev_domains=[]):
        #undo setting value in board
        self.board[idx]=0
        
        #undo updation of neighbours' domains
        for idx,dom in prev_domains:
            self.domain[idx]=set(dom)
        
    def get_MRV_idx(self):
        #return index of unassigned cell with minimum remaining domain values
        rv_heap=[]
        for i in xrange(self.s):
            idx=get_indx(i)
            
            if not self.board[idx]==0: #skip assigned cells
                continue
            
            pair = (len(self.domain[idx]),idx)
            hq.heappush(rv_heap,pair)
        min_pair = hq.heappop(rv_heap)
        return min_pair[1]
        
class Arc_Consistency():
    def __init__(self,slist):
        self.csp=Sudoku_CSP(slist)
     
    def AC3_algo(self):
        if self.AC3(): #is consistent
            return self.csp.print_board()
    
    def AC3(self):
        #returns false if inconsistency found, true otherwise
        
        arc_q=deque(self.csp.get_arcs())
        
        while not len(arc_q)==0: #while queue is not empty
            (xi,xj)=arc_q.pop()
            if self.revise(xi,xj):
                if self.csp.check_idx_D(xi):
                    return False
                xi_neighbours=self.csp.get_neighbours(xi)
                xi_neighbours.remove(xj)
                for xk in xi_neighbours:
                    arc_q.append((xk,xi))
        
        self.csp.domain_to_board()
        return True
        
    def revise(self,xi,xj):
        #returns true iff we revise the domain of xi
        revised = False
        for x in self.csp.get_idx_D(xi): #internal csp
            #check if x allowable
            if len(self.csp.get_idx_D(xj))==1:
                if x in self.csp.get_idx_D(xj):
                    self.csp.update_idx_D(xi,x)
                    revised=True
        return revised

class BTS():   
    def __init__(self,slist):
        self.csp=Sudoku_CSP(slist)
        self.previous_domains=[]
        
    def Backtracking_Search(self):
        if self.Backtrack():
            return self.csp.print_board()
        
    def Backtrack(self):
        if self.csp.is_goal():
            return True
        
        var = self.csp.get_MRV_idx() #get index of unassigned cell with minimum remaining domain values
        
        for dom_val in self.csp.get_idx_D(var):
            self.previous_domains.append(self.csp.set_idx_board(var,dom_val)) #assigning dom_val to position 'var' and storing neighbours' previous domain values
            
            if self.csp.is_game_over(lst=self.csp.get_neighbours(var)): #check consistency
                self.csp.del_idx_board(var,self.previous_domains.pop()) #removing assignment if inconsistent
                continue
                
            if self.Backtrack():
                return True
            
            self.csp.del_idx_board(var,self.previous_domains.pop()) #removing assignment
        return False
 
def main(slist):
    a = Arc_Consistency(slist)
    assignment = a.AC3_algo()
    if '0' in assignment: #board not filled
        b=BTS(slist)
        assignment=b.Backtracking_Search()
        return assignment + " BTS" 
    else:    
        return assignment + " AC3"
          
#creating Sudoku board
slist=list(sys.argv[1])

ofh=open("output.txt",'w')

ofh.write(main(slist))

ofh.close()
