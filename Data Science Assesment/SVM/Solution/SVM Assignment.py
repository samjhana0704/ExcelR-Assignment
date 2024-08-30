#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines Assignment

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ## 1. Forest Dataset 

# In[2]:


f = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\SVM\forestfires.csv")
f


# In[3]:


f.info()


# In[4]:


f.isna().sum()


# In[5]:


f.shape


# In[6]:


f1 = f.iloc[:,0:11]
forest = pd.concat([f1,f['size_category']],axis=1)
forest


# In[7]:


#Label Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
forest['month']= label_encoder.fit_transform(forest['month']) 
forest['day']= label_encoder.fit_transform(forest['day'])
forest['size_category']= label_encoder.fit_transform(forest['size_category'])


# In[8]:


forest.head()


# In[9]:


x=forest.iloc[:,0:11]
y=forest.iloc[:,-1]
x.head(7)


# In[10]:


y


# ### Visualization

# In[11]:


sns.pairplot(forest, hue='size_category')


# In[12]:


plt.figure(figsize = (14, 6));
sns.heatmap(forest.corr(), cmap='magma', annot=True, fmt=".3f")
plt.show()


# ### SVM

# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.28)


# In[14]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# ### Grid Search CV

# In[15]:


clf = SVC()

parameters = {'kernel':['rbf'],
               'gamma':[100, 75, 50, 45, 22, 5, 0.5, 0.1, 0.01, 0.0001],
               'C':[50, 35, 15, 12, 10, 6, 5, 0.1, 0.001]}

gsv = GridSearchCV(clf, param_grid = parameters, cv=10)

gsv.fit(x_train, y_train)


# In[16]:


gsv.best_params_


# In[17]:


gsv.best_score_


# In[29]:


model = SVC(C = 50, gamma = 0.0001)
model.fit(x_train , y_train)


# In[30]:


y_pred = model.predict(x_test)
y_pred


# In[31]:


acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)


# In[32]:


confusion_matrix(y_test, y_pred)


# In[22]:


print(classification_report(y_test, y_pred))


#  Inference : The Accuracy is 93.79%

# ## 2. Salary Dataset

# In[23]:


salary_train = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\SVM\SalaryData_Train.csv")
salary_train


# In[24]:


salary_train.info()


# In[25]:


salary_train.shape


# In[26]:


salary_train.isna().sum()


# In[27]:


sns.pairplot(salary_train)


# In[ ]:




