#LINEAR REGRESSION
#INPUT: training data (stored in input2.csv)
#OUTPUT: weights of model for different alphas

import numpy as np
import sys

class Linear_Regression:
    
    def __init__(self,fn=0,epsiln=0,alpha_rate=0.1):
        self.featurenum=fn #number of features (including intercept)
        self.weights=np.zeros(fn) #returns samples from standard normal distribution
        self.epsilon=epsiln #Declare convergence if cost starts decreasing by less epsilon
        self.alpha=alpha_rate #rate of convergence
    
    def scale_data(self,trainset):
        for idx,column in enumerate(trainset.T): 
            col_mean= np.mean(column)
            col_std=np.std(column)
            new_col = (column-col_mean)/col_std
            trainset.T[idx]=new_col
        return trainset
    
    def cost_R(self,trainset,labels):
        #where each row in trainset is a training example
        predictions=np.dot(trainset,self.weights)
        rss=np.sum((predictions-labels)**2)
        n=len(trainset)
        R=(0.5/n)*rss
        return R
        
    def train(self,trainset,labels,outfile="output2.csv", epsilon=0, alpha_rate=0.1):
        #where each row in trainset is an example
        global rows,cols,tot_iterations
        
        scaled_trainset=self.scale_data(trainset)
        
        #adding bias - column with ones for intercept
        one_array = np.ones(rows)
        scaled_trainset=np.hstack((one_array[:,np.newaxis],scaled_trainset))
        #bias column added, label column removed hence no change to cols
        
        #initialisations
        self.__init__(cols,epsilon,alpha_rate)
        prev_cost=150
        curr_cost=100
        
        ofh=open(outfile,'a') #OUTPUT FILE FOR ASSIGNMENT

        # while (prev_cost-curr_cost)>self.epsilon: #while decrease in cost is greater than epsilon     
        for i in xrange(tot_iterations): #FOR ASSIGNMENT   
            
            prev_cost=curr_cost #update previous cost
            
            predictions=np.dot(scaled_trainset,self.weights)
            residue=predictions-labels
            delta_weights=np.dot(scaled_trainset.T,residue)
            self.weights=self.weights-(self.alpha/float(rows))*delta_weights
            
            curr_cost=self.cost_R(scaled_trainset,labels) #update cost
        
        #OUTPUT TO FILE FOR ASSIGNMENT
        ofh.write(str(self.alpha)+','+str(tot_iterations)+',')
        ofh.write(','.join([str(w) for w in self.weights]) + '\n') 
            
        ofh.close()
        
    def predict_label(self,eg):
        if len(eg)<self.featurenum:    
            #eg does not include bias column
            eg=np.concatenate([ np.array([1]), eg ])
       
        return np.dot(self.weights,eg)
            
            

##INPUT
data=np.loadtxt(sys.argv[1], delimiter=',')

rows,cols=data.shape
feature_ubound = cols-1

#extracting training data (i.e. features)
trainegs=data[:,0:feature_ubound]

#extracting labels
lbls = data[:,feature_ubound]

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.8]
tot_iterations=100

lr=Linear_Regression()
for k in learning_rates:
    if k == learning_rates[-1]:
        tot_iterations=80
        
    lr.train(trainegs,lbls,outfile=sys.argv[2],alpha_rate=k)
    

