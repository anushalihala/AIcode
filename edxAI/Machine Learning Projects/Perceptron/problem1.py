#ALGORITHM TO TRAIN A PERCEPTRON
#INPUT: training data (stored in input1.csv)
#OUTPUT: weights for each iteration

import numpy as np
import sys
import matplotlib.pyplot as plt

class Perceptron:
    
    def __init__(self,fn=0):
        self.featurenum=fn
        self.weights=np.zeros(fn)

    def train(self,trainset,labels,outfile):
        #where each row in trainset is an example
        rows,cols=trainset.shape
        
        #adding bias - column with ones for intercept
        one_array = np.ones(rows)
        trainset=np.hstack((trainset,one_array[:,np.newaxis]))
        cols=cols+1
        
        #initialise weights
        self.__init__(cols)
        
        prev_wts=np.random.randint(10,size=cols)
        
        ofh=open(outfile,'w') #OUTPUT FILE FOR ASSIGNMENT
        while ((self.weights-prev_wts)**2).sum()!=0: #while change in weights is not zero
            prev_wts=self.weights #update previous weights
            for example,lbl in zip(trainset,labels):
                if self.classify(example)*lbl<0:
                    self.weights=self.weights+lbl*example

            ofh.write(','.join([str(w) for w in self.weights]) + '\n') #OUTPUT WEIGHTS TO FILE FOR ASSIGNMENT
            
        ofh.close()
        
    def classify(self,eg):
        if len(eg)<self.featurenum:    
            #eg does not include bias column
            eg=np.concatenate([ eg, np.array([1]) ])

        if(self.weights*eg).sum()>0:
            return 1
        else:
            return -1
            
            

##INPUT
data=np.loadtxt(sys.argv[1], delimiter=',')

#extracting training data (i.e. features x and y)
trainegs=data[:,0:2]

#extracting labels
lbls = data[:, 2]

p=Perceptron()
p.train(trainegs,lbls,sys.argv[2])

##VISUALISATIONS
# pos_egs = data[np.where(data[:,2]>0)]
# neg_egs=data[np.where(data[:,2]<0)]

# x=np.linspace(0, 16, 100)
# w1,w2,b=p.weights
# m=-(w1/w2)
# c=-(b/w2)
# y=m*x+c

# plt.xlim(0,16)
# plt.ylim(-30, 30)
# plt.plot(pos_egs[:,0],pos_egs[:,1],'bo',neg_egs[:,0],neg_egs[:,1],'ro',x,y,'-')
# plt.show()
