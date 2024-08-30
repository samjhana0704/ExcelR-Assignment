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


# ## 2. Salary Dataset

# In[2]:


s1_train = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\SVM\SalaryData_Train.csv")
s1_train


# In[3]:


salary_train = s1_train.drop_duplicates()
salary_train
import warnings
warnings.filterwarnings('ignore')


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


# ### Test Dataset:

# In[11]:


s2_test = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\SVM\SalaryData_Test.csv")
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


# In[ ]:


sns.pairplot(salary_test, hue='Salary')


# In[ ]:


plt.figure(figsize = (21, 7));
sns.heatmap(salary_test.corr(), cmap='magma', annot=True, fmt=".3f")
plt.show()


# #### Applying Train and Test split on Salary Dataset:

# In[ ]:


x_train = salary_train.iloc[:,0:13]
y_train = salary_train.iloc[:,-1]   #last column is -1


# In[ ]:


x_test = salary_test.iloc[:,0:13]
y_test = salary_test.iloc[:,-1]


# In[ ]:


#x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)


# In[ ]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# ### Using SVC :

# #### 1) kernal = rbf

# In[ ]:


model_rbf = SVC(kernel = 'rbf')
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, pred_test_rbf))(y_test, y_pred) 


# In[ ]:


confusion_matrix(y_test, pred_test_rbf)


# #### 2) kernal = linear

# In[ ]:


model_linear = SVC(kernel = 'linear')
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)
print("Accuracy:", accuracy_score(y_test, pred_test_linear))


# In[ ]:


confusion_matrix(y_test, pred_test_linear)


# #### 3) kernal = poly

# In[ ]:


model_poly = SVC(kernel = 'poly')
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)
print("Accuracy:", accuracy_score(y_test, pred_test_poly))


# In[ ]:


confusion_matrix(y_test, pred_test_poly)


# #### 4) kernal = sigmoid

# In[ ]:


model_sigmoid = SVC(kernel = 'sigmoid')
model_sigmoid.fit(x_train,y_train)
pred_test_sigmoid = model_sigmoid.predict(x_test)
print("Accuracy:", accuracy_score(y_test, pred_test_sigmoid))


# In[ ]:


confusion_matrix(y_test, pred_test_sigmoid)


# #### Using Grid Search CV :

# #### 1) kernal = rbf

# In[ ]:


class1 = SVC()
param_grid1 = [{'kernel':['rbf'], 'gamma': [50,5,0.5], 'C':[15,12,7,3,0.1,0.0001]}]
gscv1 = GridSearchCV(class1,param_grid1,cv=10)
gscv1.fit(x_train, y_train)


# In[ ]:


gscv1.best_params_, gscv1.best_score_


# In[ ]:


c1 = SVC(C=15, gamma = 0.5)
c1.fit(x_train, y_train)
y_pred1 = c1.predict(x_test) 
acc1 = accuracy_score(y_test, y_pred1)*100
print("Accuracy:", acc1)


# #### 2) kernal = linear

# In[ ]:


class2 = SVC()
param_grid2 = [{'kernel':['linear'], 'gamma': [40,5,0.5], 'C':[15,11,7,2,0.1,0.0001]}]
gscv2 = GridSearchCV(class2,param_grid2,cv=10)
gscv2.fit(x_train, y_train)


# In[ ]:


gscv2.best_params_, gscv2.best_score_


# In[ ]:


c2 = SVC(C=11, gamma = 0.5)
c2.fit(x_train, y_train)
y_pred2 = c2.predict(x_test) 
acc2 = accuracy_score(y_test, y_pred2)*100
print("Accuracy:", acc2)


# #### 3) kernal = poly

# In[ ]:


class3 = SVC()
param_grid3 = [{'kernel':['poly'], 'gamma': [45,5,0.5,0.1], 'C':[15,10,5,3,0.1,0.0001]}]
gscv3 = GridSearchCV(class3,param_grid3,cv=10)
gscv3.fit(x_train, y_train)


# In[ ]:


gscv3.best_params_, gscv3.best_score_


# In[ ]:


c3 = SVC(C=15, gamma = 0.1)
c3.fit(x_train, y_train)
y_pred3 = c3.predict(x_test) 
acc3 = accuracy_score(y_test, y_pred3)*100
print("Accuracy:", acc3)


# #### 4) kernal = sigmoid

# In[ ]:


class4 = SVC()
param_grid4 = [{'kernel':['sigmoid'], 'gamma': [50,5,0.5], 'C':[15,12,7,0.1,0.0001]}]
gscv4 = GridSearchCV(class4,param_grid4,cv=10)
gscv4.fit(x_train, y_train)


# In[ ]:


gscv4.best_params_, gscv4.best_score_


# In[ ]:


c4 = SVC(C=15, gamma = 0.5)
c4.fit(x_train, y_train)
y_pred4 = c4.predict(x_test) 
acc4 = accuracy_score(y_test, y_pred4)*100
print("Accuracy:", acc4)

