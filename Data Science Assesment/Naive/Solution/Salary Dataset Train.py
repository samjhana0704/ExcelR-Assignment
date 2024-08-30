#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Assignment

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import keras
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


# ## Salary Dataset

# ### Train Dataset:

# In[2]:


s1_train = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\Naive\SalaryData_Train.csv")
s1_train


# In[3]:


salary_train = s1_train.drop_duplicates()
salary_train
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


salary_train.describe()


# #### Label Encoding

# In[4]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
salary_train['workclass']= label_encoder.fit_transform(salary_train['workclass']) 
salary_train['education']= label_encoder.fit_transform(salary_train['education'])
salary_train['maritalstatus']= label_encoder.fit_transform(salary_train['maritalstatus'])
salary_train['occupation']= label_encoder.fit_transform(salary_train['occupation'])
salary_train['relationship']= label_encoder.fit_transform(salary_train['relationship'])
salary_train['race']= label_encoder.fit_transform(salary_train['race'])
salary_train['sex']= label_encoder.fit_transform(salary_train['sex'])
salary_train['native']= label_encoder.fit_transform(salary_train['native'])
salary_train['Salary']= label_encoder.fit_transform(salary_train['Salary'])


# In[5]:


salary_train


# In[6]:


salary_train.info()


# In[7]:


salary_train.shape


# In[8]:


salary_train.isna().sum()


# In[9]:


sns.pairplot(salary_train, hue='Salary')


# In[10]:


plt.figure(figsize = (18, 8));
sns.heatmap(salary_train.corr(), cmap='magma', annot=True, fmt=".3f")
plt.show()


# In[24]:


sns.distplot(salary_train['hoursperweek'])


# In[25]:


sns.barplot(salary_train.age,salary_train.educationno)


# In[26]:


sns.regplot(salary_train['age'],salary_train['hoursperweek'])


# ### Test Dataset:

# In[11]:


s2_test = pd.read_csv('SalaryData_Test.csv')
s2_test


# In[12]:


salary_test = s2_test.drop_duplicates()
salary_test


# #### Label Encoding

# In[13]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
salary_test['workclass']= label_encoder.fit_transform(salary_test['workclass']) 
salary_test['education']= label_encoder.fit_transform(salary_test['education'])
salary_test['maritalstatus']= label_encoder.fit_transform(salary_test['maritalstatus'])
salary_test['occupation']= label_encoder.fit_transform(salary_test['occupation'])
salary_test['relationship']= label_encoder.fit_transform(salary_test['relationship'])
salary_test['race']= label_encoder.fit_transform(salary_test['race'])
salary_test['sex']= label_encoder.fit_transform(salary_test['sex'])
salary_test['native']= label_encoder.fit_transform(salary_test['native'])
salary_test['Salary']= label_encoder.fit_transform(salary_test['Salary'])


# In[14]:


salary_test


# In[15]:


salary_test.info()


# In[16]:


salary_test.shape


# In[17]:


salary_test.isna().sum()


# In[18]:


sns.pairplot(salary_test, hue='Salary')


# In[19]:


plt.figure(figsize = (21, 7));
sns.heatmap(salary_test.corr(), cmap='magma', annot=True, fmt=".3f")
plt.show()


# #### Applying Train and Test split on Salary Dataset:

# In[20]:


x_train = salary_train.iloc[:,0:13]
y_train = salary_train.iloc[:,-1]   #last column is -1


# In[21]:


x_test = salary_test.iloc[:,0:13]
y_test = salary_test.iloc[:,-1]


# In[23]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# ## Naive Bayes Classification :

# 1. MultinomialNB
# 2. CategoricalNB
# 3. GaussianNB

# #### 1. MultinomialNB

# In[28]:


# Preparing a Multinomial naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB    # Multinomial Naive Bayes


# In[29]:


classifier_mb = MB()
classifier_mb.fit(x_train,y_train)     # Model Train


# In[30]:


# Model Accuracy on train set
train_pred_m = classifier_mb.predict(x_train)
accuracy_train_m = np.mean(train_pred_m==y_train) 


# In[31]:


# Model Accuracy on test set
test_pred_m = classifier_mb.predict(x_train)
accuracy_test_m = np.mean(test_pred_m==y_train)


# In[32]:


accuracy_train_m


# In[33]:


accuracy_test_m


# #### After applying Multinomial Naive Bayes we get 76.83% accuracy

# #### 2. CategoricalNB

# In[36]:


# Preparing a Categorical naive bayes model on training data set 
from sklearn.naive_bayes import CategoricalNB as CNB


# In[37]:


classifier_cnb = CNB()
classifier_cnb.fit(x_train,y_train)    # Model Train


# In[38]:


# Model Accuracy on train set
train_pred_cnb = classifier_cnb.predict(x_train)
accuracy_train_cnb = np.mean(train_pred_cnb==y_train) 


# In[39]:


# Model Accuracy on test set
test_pred_cnb = classifier_cnb.predict(x_train)
accuracy_test_cnb = np.mean(test_pred_cnb==y_train)


# In[40]:


accuracy_train_cnb


# In[41]:


accuracy_test_cnb


# #### After applying Categorical Naive Bayes we get 85.39% accuracy

# #### 3. GaussianNB

# In[42]:


# Preparing a Gaussian naive bayes model on training data set 
from sklearn.naive_bayes import GaussianNB as GB


# In[43]:


classifier_gb = GB()
classifier_gb.fit(x_train,y_train)


# In[44]:


# Model Accuracy on train set
train_pred_g = classifier_gb.predict(x_train)
accuracy_train_g = np.mean(train_pred_g==y_train) 


# In[45]:


# Model Accuracy on test set
test_pred_g = classifier_gb.predict(x_train)
accuracy_test_g = np.mean(test_pred_g==y_train)


# In[46]:


accuracy_train_g


# In[47]:


accuracy_test_g


# #### After applying Gaussian Naive Bayes we get 79.34% accuracy

# #### Inference: Categorical Naive Bayes gives us better Accuracy which is about 85.39%.

# In[ ]:




