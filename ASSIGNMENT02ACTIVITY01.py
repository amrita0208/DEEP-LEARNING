#!/usr/bin/env python
# coding: utf-8

# # ACTIVITY 1

# # 1)Sigmoid:
# It is also called as logistic activation function.
# f(x)=1/(1+exp(-x) the function range between (0,1)

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds
x=np.arange(-4,4,0.01)
sigmoid(x)
# Setup centered axes
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# Create and show plot
ax.plot(x,sigmoid(x)[0], color="#307EC7", linewidth=4, label="sigmoid")
ax.plot(x,sigmoid(x)[1], color="#9621E2", linewidth=4, label="derivative")
ax.legend(loc="lower right", frameon=False)
fig.show()


# # 2) tanh or Hyperbolic:
# The tanh function is just another possible functions that can be used as a nonlinear activation function between layers of a neural network. It actually shares a few things in common with the sigmoid activation function. They both look very similar. But while a sigmoid function will map input values to be between 0 and 1, Tanh will map values to be between -1 and 1.

# In[2]:


def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt
z=np.arange(-4,4,0.01)
tanh(z)[0].size,tanh(z)[1].size
# Setup centered axes
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# Create and show plot
ax.plot(z,tanh(z)[0], color="#307EC7", linewidth=3, label="tanh")
ax.plot(z,tanh(z)[1], color="#9621E2", linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
fig.show()


# # 3) RELU ACTIVATION FUNCTION
# The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance
# 
# f(x) = max(0,X)

# In[3]:


import numpy as np


# In[4]:


def relu(X):
   return np.maximum(0,X)


# In[5]:


# plot inputs and outputs
from matplotlib import pyplot
 
# rectified linear function
def relu(X):
   return np.maximum(0,X)
 
# define a series of inputs
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [relu(x) for x in series_in]
# line plot of raw inputs to rectified outputs
pyplot.plot(series_in, series_out)
pyplot.show()


# #  4)LEAKY RELU ACTIVATION
# Leaky ReLU has a small slope for negative values, instead of altogether zero. For example, leaky ReLU may have y = 0.01x when x < 0.
# 

# In[17]:



def leakyrelu(x):
   return np.maximum(0.1*x,x)


# In[18]:



# plot inputs and outputs
from matplotlib import pyplot
 
# rectified linear function
def leakyrelu(x):
   return np.maximum(0.1*x,x)
 
# define a series of inputs
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [leakyrelu(x) for x in series_in]
# line plot of raw inputs to rectified outputs
pyplot.plot(series_in, series_out)
pyplot.show()


# # 5) SOFTMAX ACTIVATION FUNCTION
# Softmax turns logits, the numeric output of the last linear layer of a multi-class classification neural network into probabilities.
# 

# In[20]:



def softmax(x):
    ''' Compute softmax values for each sets of scores in x. '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# In[23]:


x = np.linspace(-10, 10)
plt.plot(x, softmax(x))
plt.axis('tight')
plt.title('Activation Function :Softmax')
plt.show()


# # 6) BINARY STEP ACTIVATION FUNCTION
# Binary step function returns value either 0 or 1.
# 
# It returns '0' if the input is the less then zero
# 
# It returns '1' if the input is greater than zero
# 

# In[25]:


def binaryStep(x):
    ''' It returns '0' is the input is less then zero otherwise it returns one '''
    return np.heaviside(x,1)


# In[26]:


x = np.linspace(-10, 10)
plt.plot(x, binaryStep(x))
plt.axis('tight')
plt.title('Activation Function :binaryStep')
plt.show()

