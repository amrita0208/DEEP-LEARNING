#!/usr/bin/env python
# coding: utf-8

# In[1]:



#Creating Input matrix
import numpy as np


# In[2]:



values = ([2,3],[5,6],[8,9])
print("Training input values without Bias\n", values)


# In[3]:


#Adding Bias to the input values
test2 = [[-1]] * len(values)
values = np.concatenate((test2, values), axis = 1)  
print("Training input values with bias in it\n",values)


# In[4]:


#Creating random weights
m=3     #number of elements in each row of inputs
n=1 
weights = np.random.rand(m,n)*0.1 - 0.5
print("Initial random weights\n",weights)


# In[5]:



#Target values Matrix
final = ([0],[1],[1])
print("Training data target values are\n", final)


# In[7]:


#Method for updating weights
def updateWeights(weights, inputs, activation, targets):
    eta = 0.25
    weights += eta*np.dot(np.transpose(inputs), targets - activation)
    return weights


# In[8]:


#Creating Methods for Learning
def  prediction (inputs, weights, targets):
    #representing Activation function with 'ack [[]]' variable
    ack = [[0]] * len(inputs)
    for i in range(0, len(inputs)):    
        for j in range(0,len(weights)):
            ack[i] += inputs[i][j] * weights[j]
        ack[i] = np.where(ack[i]>0, 1, 0)
        #checking values with target
        if(targets[i] != ack[i]):
            weights = updateWeights(weights, inputs, ack[i], targets)
        print(ack[i])
    return weights


# In[9]:


#Training our model and extracting stable weights
iterations = 5
for temp in range(0, iterations):
    print("\nIteration ",temp+1,"\n")
    weights = prediction(values, weights, final)
    
print("\nTrained Weights\n", weights)


# In[10]:


#Testing our own data
def perceptronPredict(weights, newInput):
    ac = np.dot(newInput, weights)
    ac = np.where(ac>0, 1, 0)
    print(ac)


newInput = ([-1.0, 7, 8])
perceptronPredict(weights, newInput)

