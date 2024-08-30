#!/usr/bin/env python
# coding: utf-8

# # Random Forest Assignment

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# ## 1.Company Dataset

# In[2]:


company = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\Random Forest\Company_Data.csv")
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


company.head()


# In[10]:


company.info()


# #### Visualization

# In[11]:


sns.pairplot(company)


# In[12]:


sns.regplot('Sales','Income', data=company)  


# In[13]:


plt.figure(figsize = (8, 6));
sns.heatmap(company.corr(), cmap='magma', annot=True, fmt=".2f")


# In[14]:


company.ShelveLoc.value_counts(ascending=True).plot(kind='barh')


# In[15]:


x=company.drop(['Sales'], axis=1)
y=company[['Sales']]
x.head()


# In[16]:


y.tail()


# ### Random Forest Regressor

# In[17]:


num_trees = 70
max_features = 5


# In[18]:


kfold = KFold(n_splits=7, random_state=34, shuffle=True)
model = RandomForestRegressor(n_estimators=num_trees, max_features=max_features)


# In[19]:


results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())


# In[20]:


#The Accuracy is 67.93%


# ## 2.Fraud Check Dataset

# In[21]:


fraud = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\Random Forest\Fraud_check.csv")
fraud


# In[22]:


f_c1 = fraud.drop(['City.Population'], axis = 1)
f_c1


# In[23]:


f_c1.info()


# ##### We need to add target column for risky or not, according to the given problem statement  

# In[24]:


y1=np.empty(600, dtype=object)
i=0

for value in f_c1['Taxable.Income']:
    if value<=30000:
        y1[i]='Risky'
    else:
        y1[i]='Good'
    i=i+1    


# In[25]:


#y1


# In[26]:


t1=pd.DataFrame(y1,columns=['Target'])
t1


# In[27]:


f_c = pd.concat([f_c1,t1],axis=1)
f_c.head()


# In[28]:


f_c.isna().sum()


# In[29]:


f_c.info()


# In[30]:


f_c.corr()


# In[31]:


f_c.groupby(['Undergrad', 'Marital.Status' ,'Urban']).count()


# ##### Label Encoding

# In[32]:


label_encoder = preprocessing.LabelEncoder()
f_c['Undergrad']= label_encoder.fit_transform(f_c['Undergrad'])
f_c['Marital.Status']= label_encoder.fit_transform(f_c['Marital.Status'])
f_c['Urban']= label_encoder.fit_transform(f_c['Urban'])
f_c['Target']= label_encoder.fit_transform(f_c['Target'])


# In[33]:


f_c.head()


# In[34]:


f_c.Target.value_counts()


# In[35]:


colnames = list(f_c.columns)
colnames


# #### Visualization

# In[36]:


sns.pairplot(f_c)


# In[37]:


sns.distplot(f_c['Taxable.Income'])


# In[38]:


sns.distplot(f_c['Work.Experience'])


# In[39]:


plt.figure(figsize = (8, 6));
sns.heatmap(f_c.corr(), cmap='magma', annot=True, fmt=".2f")


# In[40]:


sns.scatterplot(x = 'Taxable.Income', y = 'Work.Experience', data = f_c)


# In[41]:


x1=f_c.iloc[:,0:5]
y1=f_c[['Target']]
x1.head()


# In[42]:


y1.tail()


# ### Random Forest Classification

# In[43]:


num_trees = 82
max_features = 4


# In[44]:


kfold = KFold(n_splits=8, random_state=27, shuffle=True)
model2 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


# In[45]:


results = cross_val_score(model2, x1, y1, cv=kfold)
print(results.mean())


# In[46]:


# The Accuracy for this dataset is 99.83%


# ### Random Forest Regression

# In[47]:


num_trees = 65
max_features = 7


# In[48]:


kfold = KFold(n_splits=7, random_state=14, shuffle=True)
model3 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


# In[49]:


results = cross_val_score(model3, x1, y1, cv=kfold)
print(results.mean())


# In[50]:


#The Accuracy is 99.83%

