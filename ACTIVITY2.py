#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Location of dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=names)


# In[7]:


# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]

# Assign data from first fifth columns to y variable
y = irisdata.select_dtypes(include=[object])


# In[8]:


irisdata.head()


# In[10]:


y.head()


# In[16]:


y.Class.unique()


# In[17]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)


# In[19]:


y.Class.unique()


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# # FEATURE SCALING

# In[23]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # TRAINING AND PREDICTION

# In[25]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())


# In[27]:


predictions = mlp.predict(X_test)


# # EVALUATION OF ALGORITHM

# In[28]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

