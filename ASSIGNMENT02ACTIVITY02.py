#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# In[3]:


#reading the image 
img=cv2.imread("C:/Users/amrit/Downloads/img.jpg")
#displaying original image
plt.imshow(img)


# In[4]:


#resizing the image square dimension
img=cv2.resize(img,(500,500))
#defining the filter of 3 X 3 size and applying it first
#filter1=np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)
#defining the filter of 5 X 5 size and comapre it
filter1=np.array([(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1)])*(1/25)
print(filter1)


# In[6]:



C=img.shape
F=filter1.shape
#converting the colouerd  image into grayscale image or (Binary conversion)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.title("Before_coverting")
plt.imshow(img)


# In[7]:


plt.title("The original grayscaled image")
plt.imshow(img_gray)


# In[8]:


print(img_gray)


# In[9]:


print(img_gray)
print(type(img_gray))
img_gray2=img_gray
print(img_gray2)
print(type(img_gray2))


# # BOX FILTER
# The Box Filter operation is similar to the averaging blur operation; it applies a bilateral image to a filter. Here, you can choose whether the box should be normalized or not
# 

# In[10]:


#(2A) Applying Box filter of 3 X 3 and 5 X 5 and comapring it.

def pro_sum(m,n):
    sum=0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            sum=sum+(m[i,j]*n[i,j])
    return math.ceil(sum)
X1=np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)
X2=np.array([(50,50,49),(51,50,50),(51,50,50)])
print(pro_sum(X1,X2))
for i in range(C[0]):
    for j in range(C[1]):
        k = img_gray[i:i+F[0],j:j+F[1]]#slicing the image in the form of multiple filter dimension 
        l = pro_sum(k,filter1)
        img_gray2[i][j]=l
print(img_gray2)   


# In[9]:


plt.title("The convoluted image is ")
plt.imshow(img_gray2)


# In[10]:


#(2B)Applying Box filter of 3 X 3 and 5 X 5 and comapre it.HERE THE STRIDE=2
img_gray3=img_gray
import math
def pro_sum(m,n):
    sum=0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            sum=sum+(m[i,j]*n[i,j])
    return math.ceil(sum)

for i in range(C[0]):
    for j in range(C[1]):
        k = img_gray[i:i+F[0],j:j+F[1]]#slicing the image in the form of multiple filter dimension 
        l = pro_sum(k,filter1)
        img_gray3[i][j]=l
        j+=1 #here the stride is 2 or filter jumps by 2 pixals intead of 1
    i+=1
print(img_gray3)        
plt.title("The 2 stride convoluted image is ")
plt.imshow(img_gray3)


# # ZERO PADDING

# In[11]:


#(2C) Apply zero padding before applying Box filter of 3 X 3 and 5 X 5 and comapre it
A=C[0]+F[0]-1
B=C[1]+F[1]-1
Y= np.zeros((A,B))#creating the image of all zero intensity pixals
print(Y)
print(Y.shape)
Z=Y


# In[14]:



#fitting the input image in the centre of zero intensity pixals image Y to form zero padeded image Z
for i in range(C[0]):
    for j in range(C[1]):
        m=np.int((F[0]-1)/2)
        n=np.int((F[1]-1)/2)
        Z[i+m,j+n]=img_gray[i,j]
print("The pixal values after zero padding is ")
print(Z)
print(Z.shape)
print(Z[0,0],Z[1,1])


# In[15]:


img_gray4=img_gray
import math
def pro_sum(m,n):
    sum=0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            sum=sum+(m[i,j]*n[i,j])
    return math.ceil(sum)

for i in range(C[0]):
    for j in range(C[1]):
        k = Z[i:i+F[0],j:j+F[1]]#slicing the image in the form of multiple filter dimension 
        l = pro_sum(k,filter1)
        img_gray3[i][j]=l

print(img_gray3)        
plt.title("The Zero padded convoluted image is ")
plt.imshow(img_gray4)


# # ENTROPY
# Entropy is a measure of image information content, which is interpreted as the average uncertainty of information source. In Image, Entropy is defined as corresponding states of intensity level which individual pixels can adapt
# 

# In[16]:


#Entropy of the input image
import skimage.measure    
entropy = skimage.measure.shannon_entropy(img_gray)
print(entropy)

