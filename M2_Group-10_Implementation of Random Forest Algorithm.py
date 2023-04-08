#!/usr/bin/env python
# coding: utf-8

# # Implementation of Random Forest Algorithm
# 
# #### Group-10
# #### Aryan Pandey (1906465)
# #### Hari Om (1906476)
# #### Samim Hossain Mondal (1906500)
# #### Souhardya Pal (1906511)
# #### Tapabrata Roy (1906522)

# In[1]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the given data set
iris = pd.read_csv("C:\\Users\\KIIT\\Documents\\3rd Year\\6th SEM\\Machine Learning\\M2\\iris.csv")
iris


# In[3]:


#checking the datatypes of the dataset
iris.dtypes


# In[4]:


#splitting the dataset into two parts; one part contains the float variables and another one takes only the categoricl variable
x = iris.iloc[:,:-1].values
y = iris.iloc[:, -1].values


# In[5]:


print(x)


# In[6]:


print(y)


# In[7]:


print("Length of x =",len(x))
print("Length of y =",len(y))


# In[8]:


#Encoding the categorical dependent variable
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y = label.fit_transform(y)


# In[9]:


print(y)


# In[10]:


#splitting into train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 15)


# In[11]:


print(x_train)


# In[12]:


print(x_test)


# In[13]:


print(y_train)


# In[14]:


print(y_test)


# In[15]:


print("Length of x_train =",len(x_train))
print("Length of x_test =",len(x_test))
print("Length of y_train =",len(y_train))
print("Length of y_test =",len(y_test))


# In[16]:


#defining the random forest classifier
from sklearn.ensemble import RandomForestClassifier
#create classifier object
clf = RandomForestClassifier(n_estimators = 5, criterion = 'gini', random_state = 15)


# In[17]:


#fit the classifier and training the random forest classifier
clf.fit(x_train, y_train)


# In[18]:


#predicting the test result
predict = clf.predict(x_test)
print(predict)


# In[19]:


predict_vertical = predict.reshape(len(predict),1)
print(predict_vertical)


# In[20]:


original_vertical = y_test.reshape(len(y_test),1)
print(original_vertical)


# In[21]:


original_predict = np.concatenate((original_vertical, predict_vertical), axis = 1)
print(original_predict)


# In[22]:


#plotting confusion matrix
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, predict), annot = True)
plt.xlabel("Predicted Label")
plt.ylabel("Original Label")
plt.title("Confusion Matrix")
plt.show()


# In[23]:


#Finding the classifier Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predict)


# In[24]:


#evaluating the classifier on new data
prediction = clf.predict([[5, 3, 1.6, 0.2]])
print(prediction)

