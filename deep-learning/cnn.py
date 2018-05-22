#TODO:
#output comments
#which need to be np
# output function
# cost function
# train function -> error from labels ->propagate back, fixed iter + check cost
# fc bp -> update weights, return errors for all m
# relu bp 
# con bp


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
        self.bias=  np.zeros(self.filter_num)
        
        # TESTING
        # self.filter_num = 4
        # self.filter_size = 3
        # self.filters = np.array([ [[1,0,0],[0,0,0],[0,0,0]],[[0,0,1],[0,0,0],[0,0,0]],[[1,0,0],[0,0,0],[0,0,0]],[[0,0,1],[0,0,0],[0,0,0]] ])

        self.stride=in_stride
        self.padding=in_padding
        
    def forward_pass(self, input_data):
        #INPUTS
        #input_data: numpy array of dimensions [m,w,w], where there are m samples of height and width w
        #OUTPUT
        #resulting layers after applying filters
        
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
                       
                        output[k,f,i,j] = np.sum(input_data[k, i_start_index:i_end_index ,  j_start_index:j_end_index] * self.filters[f]) + self.bias[f]
                       
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
        #OUTPUT
        #result of applying f(x) to all input values
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
        
        self.W=np.random.rand(i,j+1) # Weights of network)
    
    def forward_pass(self, input_data):
        #INPUTS
        #input_data: numpy array of dimensions [m,w,w], where there are m samples of height,width = w
        #OUTPUT
        #values of output layer
        
        #ensuring input is numpy array
        if(not isinstance(input_data,np.ndarray)):   
            input_data=np.array(input_data)
            
        bias = np.ones(1)
        input_vectors=[]
        for sample in input_data:
            input_vectors.append(np.append(bias,sample.ravel()))
        
        input_vectors=np.array(input_vectors)
        
        output= np.dot(self.W,input_vectors.T)
        
        np.apply_along_axis(self.sigmoid,0,output)
        
        return output
        
    def backward_pass(self):
        pass
    
    def sigmoid(self, x):
        exp_val=np.exp(x)
        return exp_val/(1+exp_val)
        
    def sigmoid_gradient(self,x):
        sigmoid_val=self.sigmoid(x)
        return sigmoid_val*(1-sigmoid_val)
        

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
        
        #if forward pass computation of convolution layer unsuccessful
        if(isinstance(a_conv,int)):
            return
        
        a_relu=self.relu1.forward_pass(a_conv)
        a_fc=self.fc1.forward_pass(a_relu)
        
        output=a_fc.argmax(0)
        output=output+1
        
        # TESTING
        print(a_conv)
        print(a_relu)
        print(a_fc)
        print(output)
        
        return output
        
    def cost(self, output, labels):
        #INPUTS
        #output: numpy array of dimensions [m, 10] for m samples; values output by CNN
        #labels: numpy array of dimensions [m, 10] for m samples; labels for training data
        
        #ensuring inputs are numpy arrays
        if(not isinstance(output,np.ndarray)):   
            output=np.array(output)
        if(not isinstance(labels,np.ndarray)):   
            labels=np.array(labels)
            
        #checking for single sample values
        if(len(output.shape)==2):
            m=len(output)
        else:
            m=1
        
        return (1/m)*np.sum(-labels*np.log(output) - (1-labels)*np.log(1-output))
    
    def train(self):
        pass

        
def augment_output(simple_labels):
    #INPUTS
    #simple_labels: array of dimensions [m]; values range from 1 to 10; label x indicates output neuron x = 1 and all other output neurons = 0
    #OUTPUT
    #array of dimensions [m, 10] where for each i=1,2..m, [i,x]=1 and [i,not x]=0
    
    m=len(simple_labels)
    augmented_labels=np.zeros([m,10])
    for i in range(m):
        ith_label=simple_labels[i]
        augmented_labels[i,ith_label-1]=1
        
    return augmented_labels

#PARAMETERS
# number_of_filters = 4
# filter_dim = 3
# input_dim = 32
# padding = 1
# stride = 2

# TESTING
# inputstr="1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2"
# inputlist=inputstr.split(',')
# inputlist=[int(i) for i in inputlist]
# inputlist=inputlist*32
# inputvect=np.array(inputlist)
# inputdata=inputvect.reshape(32,32)
# inputdata2=np.array([inputdata,inputdata])
# print(inputdata2)

cnet=CNN1()
# print(cnet.compute_output(inputdata2))

# a=[[1,0,0],[0,1,0]]
# b=[[0.2,0.01,0.01],[0.01,0.9,0.4]]
# print(cnet.cost(b,a))
