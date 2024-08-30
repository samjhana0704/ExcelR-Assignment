#!/usr/bin/env python
# coding: utf-8

# # Decision Tree  Assignment

# In[2]:


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


# ## 2.Fraud Check Dataset

# In[3]:


fraud = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\Decision Tree\Fraud_check.csv")
fraud


# In[4]:


f_c1 = fraud.drop(['City.Population'], axis = 1)
f_c1


# In[5]:


f_c1.info()


# ##### We need to add target column for risky or not, according to the given problem statement  

# In[6]:


y1=np.empty(600, dtype=object)
i=0

for value in f_c1['Taxable.Income']:
    if value<=30000:
        y1[i]='Risky'
    else:
        y1[i]='Good'
    i=i+1    


# In[7]:


y1


# In[8]:


t1=pd.DataFrame(y1,columns=['Target'])
t1


# In[9]:


f_c = pd.concat([f_c1,t1],axis=1)
f_c.head()


# In[10]:


f_c.isna().sum()


# In[11]:


f_c.info()


# In[12]:


f_c.corr()


# In[13]:


f_c.groupby(['Undergrad', 'Marital.Status' ,'Urban']).count()


# ##### Label Encoding

# In[14]:


label_encoder = preprocessing.LabelEncoder()
f_c['Undergrad']= label_encoder.fit_transform(f_c['Undergrad'])
f_c['Marital.Status']= label_encoder.fit_transform(f_c['Marital.Status'])
f_c['Urban']= label_encoder.fit_transform(f_c['Urban'])
f_c['Target']= label_encoder.fit_transform(f_c['Target'])


# In[15]:


f_c.head()


# In[16]:


f_c.Target.value_counts()


# In[17]:


colnames = list(f_c.columns)
colnames


# #### Visualization

# In[18]:


sns.pairplot(f_c)


# In[20]:


sns.distplot(f_c['Taxable.Income'])


# In[21]:


sns.distplot(f_c['Work.Experience'])


# In[22]:


plt.figure(figsize = (8, 6));
sns.heatmap(f_c.corr(), cmap='magma', annot=True, fmt=".2f")


# In[23]:


sns.scatterplot(x = 'Taxable.Income', y = 'Work.Experience', data = f_c)


# In[24]:


x=f_c.iloc[:,0:5]
y=f_c[['Target']]
x.head()


# In[25]:


y.tail()


# ### Building Decision Tree Classifier using Entropy Criteria

# In[26]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=32)


# In[27]:


model2 = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model2.fit(x_train,y_train)


# In[28]:


tree.plot_tree(model2);                            #PLot the decision tree


# In[29]:


fn=['Undergrad', 'Marital.Status', 'Taxable.Income', 'Work.Experience', 'Urban']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,2), dpi=300)
tree.plot_tree(model2,
               feature_names = fn, 
               #class_names=cn,
               filled = True);


# In[30]:


preds2 = model2.predict(x_test)
pd.Series(preds2).value_counts()


# In[31]:


preds2


# In[32]:


y_test


# In[33]:


y_test2 = y_test.to_numpy()
y_test2 = np.reshape(y_test2, 180)
y_test2


# In[34]:


pd.crosstab(y_test2,preds2) # getting the 2 way table to understand the correct and wrong predictions


# ### Building Decision Tree Classifier (CART) using Gini Criteria

# In[35]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.22,random_state=18)


# In[36]:


from sklearn.tree import DecisionTreeClassifier
model2_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[37]:


model2_gini.fit(x_train, y_train)


# In[50]:


y_test


# In[39]:


y_test3 = y_test.to_numpy()
y_test3 = np.reshape(y_test3, 132)
y_test3


# In[40]:


#Prediction and computing the accuracy
pred=model2.predict(x_test)
np.mean(pred==y_test3)


# #### Decision Tree Regression Example

# In[41]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[51]:


array = f_c.values
X = array[:,0:5]
y = array[:,-1]


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)


# In[53]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[54]:


model.score(X_test,y_test)           #Accuracy


# In[ ]:




