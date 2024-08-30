#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbour (KNN) Assignment

# In[1]:


import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ## 1. Glass Dataset

# In[2]:


glass = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\KNN\glass.csv")
glass


# In[3]:


glass.shape


# In[4]:


glass.info()


# In[5]:


glass.describe()


# In[6]:


glass.isna().sum()


# ### Visualization

# In[7]:


sns.pairplot(glass, hue='Type')


# In[8]:


plt.figure(figsize = (9, 6));
sns.heatmap(glass.corr(), cmap='magma', annot=True, fmt='.3f')
plt.show()


# In[9]:


sns.violinplot(glass['RI'],glass['Na'])


# ### KNN Model Building

# In[10]:


X = glass.iloc[:, 0:9]
Y = glass.iloc[:, -1]
X


# In[11]:


model = KNeighborsClassifier(n_neighbors = 12)


# In[12]:


model.fit(X,Y)


# In[13]:


kfold = KFold(n_splits=8)
results = cross_val_score(model, X, Y, cv = kfold)


# In[14]:


results


# In[15]:


results.mean()


# In[16]:


model.predict([[1.51651,14.38,0.00,1.94,73.61,0.00,8.48,1.57,0.0]])


# ### Grid Search for Algorithm Tuning

# In[17]:


n_neighbors = list(range(1,33))
parameters = {'n_neighbors' : n_neighbors}


# In[18]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator = model, param_grid = parameters)
grid.fit(X, Y)


# In[19]:


print(grid.best_score_)
print(grid.best_params_)


# ### Visualizing the CV results

# In[20]:


k_range = range(1, 57)
k_scores = []

# use iteration to caclulator different k in models
# then return the average accuracy based on the cross validation

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, Y, cv=4)
    k_scores.append(scores.mean())


# In[21]:


plt.figure(figsize=(16,6))
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[22]:


k_scores


# In[23]:


glass_accuracy = pd.DataFrame({'Value of k' : range(1,57),
                            'Accuracy' : k_scores})
glass_accuracy


# In[24]:


glass_accuracy.sort_values('Accuracy', ascending = False)


# ## 2. Zoo Dataset

# In[25]:


zoo = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\KNN\Zoo.csv")
zoo


# In[26]:


zoo.shape


# In[27]:


zoo.info()


# In[28]:


zoo.describe()


# In[29]:


zoo.isna().sum()


# In[30]:


zoo['animal name'].unique()


# In[31]:


zu=zoo['animal name'].astype('category')
zu.unique


# ### Visualization

# In[32]:


sns.pairplot(zoo)


# In[33]:


plt.figure(figsize = (20, 8));
sns.heatmap(zoo.corr(),cmap='magma', annot=True, fmt='.3f')
plt.show()


# In[34]:


sns.barplot(zoo['legs'],zoo['catsize'])


# ### KNN Model Building

# In[35]:


#from sklearn import preprocessing
#label_encoder = preprocessing.LabelEncoder()
#zoo['animal name']= label_encoder.fit_transform(zoo['animal name']) 


# In[36]:


X = zoo.iloc[:, 1:17]
Y = zoo.iloc[:, -1]
X


# In[37]:


Y


# In[38]:


model2 = KNeighborsClassifier(n_neighbors = 17)


# In[39]:


model2.fit(X,Y)


# In[40]:


kfold2 = KFold(n_splits=12)
results2 = cross_val_score(model2, X, Y, cv = kfold2)


# In[41]:


results2


# In[42]:


results2.mean()


# ##### Grid Search for Algorithm Tuning

# In[43]:


n_neighbors2 = list(range(1,40))
parameters2 = {'n_neighbors' : n_neighbors2}


# In[44]:


model2 = KNeighborsClassifier()
grid2 = GridSearchCV(estimator = model2, param_grid = parameters2)
grid2.fit(X, Y)


# In[45]:


print(grid2.best_score_)
print(grid2.best_params_)


# ### Visualizing the CV results

# In[46]:


k_range = range(1,35)
k_scores = []


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, Y, cv=4)
    k_scores.append(scores.mean())


# In[47]:


plt.figure(figsize=(16,6))
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[48]:


k_scores


# In[49]:


zoo_accuracy = pd.DataFrame({'Value of k' : range(1,35),
                            'Accuracy' : k_scores})
zoo_accuracy


# In[50]:


zoo_accuracy.sort_values('Accuracy', ascending = False)


# In[ ]:




