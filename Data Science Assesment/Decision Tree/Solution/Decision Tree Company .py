#!/usr/bin/env python
# coding: utf-8

# # Decision Tree  Assignment

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')


# ## 1.Company Dataset

# In[2]:


company = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\Decision Tree\Company_Data.csv")
company


# In[3]:


company.isna().sum()


# In[4]:


company.info()


# In[5]:


company.corr()


# In[6]:


#company.Sales.unique()


# In[7]:


company.groupby(['ShelveLoc','Urban','US']).count()


# ##### Label Encoding

# In[8]:


label_encoder = preprocessing.LabelEncoder()
company['ShelveLoc']= label_encoder.fit_transform(company['ShelveLoc']) 
company['Urban']= label_encoder.fit_transform(company['Urban'])
company['US']= label_encoder.fit_transform(company['US'])


# In[9]:


company['ShelveLoc']=company['ShelveLoc'].astype('category')
company['Urban']=company['Urban'].astype('category')
company['US']=company['US'].astype('category')


# In[10]:


company.head()


# In[11]:


type(company.ShelveLoc)


# In[12]:


company.info()


# In[13]:


colnames = list(company.columns)
colnames


# #### Visualization

# In[14]:


sns.pairplot(company)


# In[15]:


sns.regplot('Sales','Income', data=company)  


# In[16]:


plt.figure(figsize = (8, 6));
sns.heatmap(company.corr(), cmap='magma', annot=True, fmt=".2f")


# In[17]:


company.ShelveLoc.value_counts(ascending=True).plot(kind='barh')


# In[18]:


#x=company.drop(['Sales'], axis=1)
#y=company[['Sales']]
#x.head()


# In[19]:


import ppscore as pps
pps.matrix(company)        #calculate the whole PPS matrix


# In[20]:


pps.score(company, "Sales", "Income")


# #### For continous variable we can't use classifier i.e. we are going to use Regressor for this dataset

# #### Decision Tree Regression

# In[21]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

x=company.drop(['Sales'], axis=1)
y=company[['Sales']]
# In[22]:


array = company.values
X = array[:,1:11]
y = array[:,0]
y


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[24]:


model0 = DecisionTreeRegressor()
model0.fit(X_train, y_train)


# In[25]:


model0.score(X_test,y_test)           #Accuracy


# In[ ]:




