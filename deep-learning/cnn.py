import numpy as np
import pdb
  

class Convolution:
    # 2D Convolution
    def __init__(self, filter_parameters, in_stride, in_padding):
        #INPUTS
        #filter_parameters: [number of filters, filter height or width]
        #in_stride: stride
        #in_stride: padding 
        
        self.filter_num = filter_parameters[0]
        self.filter_size = filter_parameters[1]
        self.filters = np.random.rand(filter_parameters[0],filter_parameters[1],filter_parameters[1])
        
        # TESTING
        # self.filter_num = 4
        # self.filter_size = 3
        # self.filters = np.array([ [[1,0,0],[0,0,0],[0,0,0]],[[0,0,1],[0,0,0],[0,0,0]],[[1,0,0],[0,0,0],[0,0,0]],[[0,0,1],[0,0,0],[0,0,0]] ])

        self.stride=in_stride
        self.padding=in_padding
        
    def forward_pass(self, input_data):
        #INPUTS
        #input_data: numpy array of dimensions [m,w,w], where there are m samples of height and width w
        
        #ensuring input is numpy array
        if(not isinstance(input_data,np.ndarray)):   
            input_data=np.array(input_data)
            
        #checking for single sample input
        if(len(input_data.shape)==2):
            input_data=np.array([input_data])  
            
        m=input_data.shape[0]
        w=input_data.shape[1]
        
        if(w<self.filter_size):
            return -1 #ERROR
        
        o = int(((w - self.filter_size + 2*self.padding)/self.stride) + 1) #output height or width
        output = np.zeros([m,self.filter_num,o,o])                         #initialising output with zeros
        
        #Adding padding
        input_data=np.pad(input_data,[(0,0),(self.padding,self.padding),(self.padding,self.padding)],'constant')
        
        for k in range(m):                          #for each input sample
            for f in range(self.filter_num):        #for each filter
                for i in range(o): 
                    i_start_index=i*self.stride
                    i_end_index=i_start_index+self.filter_size
                    
                    for j in range(o):  
                        j_start_index=j*self.stride
                        j_end_index=j_start_index+self.filter_size
                       
                        output[k,f,i,j] = np.sum(input_data[k, i_start_index:i_end_index ,  j_start_index:j_end_index] * self.filters[f])
                       
        return output
    
    def backward_pass(self):
        pass

class ReLU:
    # ReLU layer applies function f(x)=max(0,x) to input
    def __init__(self):
        pass
        
    def forward_pass(self, input_data):
        #INPUTS
        #input_data: numpy array of dimensions [d,w,w], of depth = d, and height,width = w
        vmax=np.vectorize(max)
        return vmax(input_data,0)
    
    def backward_pass(self):
        pass
        
class FC:
    # Fully connected neural network with no hidden layer
    def __init__(self,i,j):
        # INPUTS
        # i=number of rows/number of neurons in output layer
        # j=number of columns/number of neurons in input layer
        
        self.W=np.random.rand(i,j) # Weights of network
        
        # TESTUBG
        # self.W=np.ones([i,j])
    
    def forward_pass(self, input_data):
        #INPUTS
        #input_data: numpy array of dimensions [m,w,w], where there are m samples of height,width = w
        
        #ensuring input is numpy array
        if(not isinstance(input_data,np.ndarray)):   
            input_data=np.array(input_data)
            
        input_vectors=[]
        for sample in input_data:
            input_vectors.append(sample.ravel())
        
        input_vectors=np.array(input_vectors)
        
        output= np.dot(self.W,input_vectors.T)
        return output
        
    def backward_pass(self):
        pass

class CNN1:
    def __init__(self):
        #CNN with the following architecture:
        #Input - 32x32 matrix
        #Convolution layer with 4 filters ... Filter size = (3x3), Output of layer = (16x16x4)
        #ReLU layer
        #Fully connected layer with no hidden layers, with 1024 input neurons and 10 output neurons
        #Output  - 10x1 vector
        
        self.conv1=Convolution([4,3],2,1)
        self.relu1=ReLU()
        self.fc1=FC(10,1024)
        
    def compute_output(self, input_samples):
        #INPUTS
        #input_samples: array of dimensions [m,w,w], where there are m samples of height and width w
        
        # calculating activations for each layer
        a_conv = self.conv1.forward_pass(input_samples)
        a_relu=self.relu1.forward_pass(a_conv)
        a_fc=self.fc1.forward_pass(a_relu)
        
        output=a_fc.argmax(0)
        output=output+1
        
        #TESTING
        print(a_conv)
        print(a_relu)
        print(a_fc)
        print(output)
        
        return output
    
    def train(self):
        pass

#PARAMETERS
number_of_filters = 4
filter_dim = 3
input_dim = 32
padding = 1
stride = 2

#TEST
# testinput=[[[1,2,3],[4,5,6],[7,8,9]],[[1,1,1],[1,1,1],[1,1,1]]]

# conv=Convolution([1,2],1,1)
# print(conv.forward_pass(testinput))

# testinput2=[[[1,2,3],[-4,-5,-6],[7,8,9]],[[-1,-1,-1],[1,1,1],[1,1,1]]]
# r=ReLU()
# print(r.forward_pass(testinput2))
# fcl=FC(10,9)
# print(fcl.forward_pass(testinput2))
###############

inputstr="1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2"
inputlist=inputstr.split(',')
inputlist=[int(i) for i in inputlist]
inputlist=inputlist*32
inputvect=np.array(inputlist)
inputdata=inputvect.reshape(32,32)
inputdata2=np.array([inputdata,inputdata])
print(inputdata2)

cnet=CNN1()
cnet.compute_output(inputdata2)
