#!/usr/bin/env python
# coding: utf-8

# # Neural Network  Assignment

# In[35]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import adam_v2
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, KFold


# ## 1. Forest Fires Dataset 

# In[2]:


f = pd.read_csv(r"C:\Users\gupta\Downloads\Data Science Assesment\Neural net\forestfires.csv)
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


plt.figure(figsize = (16, 8));
sns.boxenplot(x = 'temp', y = 'wind', data = f1)


# In[8]:


sns.countplot(f1['month'], palette="Set1")


# In[9]:


plt.figure(figsize = (16, 8));
sns.violinplot(x = 'DMC', y = 'DC', data = f1)


# In[10]:


#Label Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
forest['month']= label_encoder.fit_transform(forest['month']) 
forest['day']= label_encoder.fit_transform(forest['day'])
forest['size_category']= label_encoder.fit_transform(forest['size_category'])


# In[11]:


forest.head()


# In[12]:


x=forest.iloc[:,0:11]
y=forest.iloc[:,-1]
x.head(7)


# In[13]:


y


# ### Visualization

# In[14]:


sns.pairplot(forest, hue='size_category')


# In[15]:


plt.figure(figsize = (14, 6));
sns.heatmap(forest.corr(), cmap='magma', annot=True, fmt=".3f")
plt.show()


# ## Neural Network

# In[16]:


np.random.seed(7)

# split into input (X) and output (Y) variables
X = forest.iloc[:,0:11]
Y = forest.iloc[:,-1]


# In[17]:


Y


# ### 1. Batch Size and Epochs

# In[18]:


# create model
model = Sequential()
model.add(Dense(14, input_dim=11,  activation='relu')) #1st layer
model.add(Dense(11,  activation='relu')) #2nd layer
model.add(Dense(1, activation='sigmoid')) #3rd layer or op layer


# In[19]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[20]:


# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=250, batch_size=10)


# In[21]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[22]:


model.metrics_names


# In[23]:


scores


# In[24]:


# Visualize training history

# list all data in history
history.history.keys()


# In[25]:


# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[26]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## Neural Network Hyper Parameter :

# In[27]:


# Standardization

a = StandardScaler()
a.fit(X)
X_standardized = a.transform(X)


# In[28]:


pd.DataFrame(X_standardized).describe()


# ##### Create model using function:

# In[29]:


def create_model():
    model = Sequential()
    model.add(Dense(15, input_dim=11, init='uniform', activation='relu'))
    model.add(Dense(11, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    adam=Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# ##### Create the model:

# In[30]:


model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)


# #### Tuning of Hyperparameters using different techniques
#     1. Batch Size and Epochs
#     2. Learning rate and Drop out rate
#     3. Activation Function and Kernel Initializer
#     4. Number of Neurons in Activation layer
#     5. Training model with optimum values of Hyperparameters    

# ### 2. Learning rate and Drop out rate

# In[31]:


from keras.layers import Dropout

# Defining the model

def create_model(learning_rate, dropout_rate):
    model = Sequential()
    model.add(Dense(11,input_dim = 11,kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(14,input_dim = 11,kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation = 'sigmoid'))
    
    adam = adam_v2.Adam(lr = learning_rate)

    model.compile(loss = 'binary_crossentropy',
                  optimizer = adam,
                  metrics = ['accuracy'])
    
    return model


# In[32]:


# Create the model

model = KerasClassifier(build_fn = create_model,
                        verbose = 0,
                        batch_size = 40,
                        epochs = 50)


# In[33]:


# Define the grid search parameters

learning_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.0, 0.1, 0.2]

# Make a dictionary of the grid search parameters

param_grids = dict(learning_rate = learning_rate,
                   dropout_rate = dropout_rate)


# In[36]:


# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,
                    param_grid = param_grids,
                    cv = KFold(),
                    verbose = 10)

grid_result = grid.fit(X_standardized,Y)


# In[37]:


# Summarize the results
print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param))


# Inference: The best score is 97.28 % , using 'dropout_rate': 0.2 as 'learning_rate': 0.1

# #### 3. Activation Function and Kernel Initializer

# In[38]:


# Defining the model

def create_model(activation_function,init):
    model = Sequential()
    model.add(Dense(11,input_dim = 11,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(0.2))
    model.add(Dense(14,input_dim = 11,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation = 'sigmoid'))
    
    adam = adam_v2.Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model


# In[39]:


# Create the model

model = KerasClassifier(build_fn = create_model,
                        verbose = 0,
                        batch_size = 40,
                        epochs = 50)


# In[40]:


# Define the grid search parameters
activation_function = ['softmax', 'relu', 'tanh', 'linear']
init = ['uniform', 'normal', 'zero']

# Make a dictionary of the grid search parameters
param_grids = dict(activation_function = activation_function,
                   init = init)


# In[41]:


# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,
                    param_grid = param_grids,
                    cv = KFold(),
                    verbose = 10)

grid_result = grid.fit(X_standardized,Y)


# In[42]:


# Summarize the results
print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param))


# Inference: The best score is 92.25 % , using 'activation_function': 'tanh' as 'init': 'uniform'

# ### 4. Number of Neurons in activation layer

# In[43]:


# Defining the model

def create_model(neuron1,neuron2):
    model = Sequential()
    model.add(Dense(neuron1,input_dim = 11,kernel_initializer = 'uniform',activation = 'linear'))
    model.add(Dropout(0.2))
    model.add(Dense(neuron2,input_dim = neuron1,kernel_initializer = 'uniform',activation = 'linear'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation = 'sigmoid'))
    
    adam = adam_v2.Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = adam,
                  metrics = ['accuracy'])
    return model


# In[44]:


# Create the model

model = KerasClassifier(build_fn = create_model,
                        verbose = 0,
                        batch_size = 40,
                        epochs = 50)


# In[45]:


# Define the grid search parameters

neuron1 = [4,8,16]
neuron2 = [2,4,8]

# Make a dictionary of the grid search parameters

param_grids = dict(neuron1 = neuron1,
                   neuron2 = neuron2)


# In[46]:


# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,
                    param_grid = param_grids,
                    cv = KFold(),
                    verbose = 10)

grid_result = grid.fit(X_standardized,Y)


# In[47]:


# Summarize the results
print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param))


# Inference: The best score is 91.28 % , using 'neuron1': 16 and 'neuron2': 8

# ## 2. Gas Turbine Dataset

# In[48]:


turbine1 = pd.read_csv('gas_turbines.csv')
turbine1


# In[49]:


turbine = turbine1.drop_duplicates()    #cleaning duplicates
turbine.shape


# In[50]:


turbine.isna().sum()


# In[51]:


turbine.info()


# In[52]:


turbine.describe()


# ### Visualization

# In[53]:


sns.pairplot(turbine)


# In[54]:


plt.figure(figsize = (14, 6));
sns.heatmap(turbine.corr(), cmap='magma', annot=True, fmt=".3f")
plt.show()


# In[55]:


f, axes = plt.subplots(2, 2, figsize=(12,8))

sns.violinplot(x = 'AT', y = 'TEY', data = turbine, scatter_kws={'alpha':0.6}, ax = axes[0,0])
sns.violinplot(x = 'AP', y = 'TEY', data = turbine, scatter_kws={'alpha':0.6}, ax = axes[0,1])
sns.violinplot(x = 'AH', y = 'TEY', data = turbine, scatter_kws={'alpha':0.6}, ax = axes[1,0])
sns.violinplot(x = 'AFDP', y = 'TEY', data = turbine, scatter_kws={'alpha':0.6}, ax = axes[1,1])


# In[56]:


f, axes = plt.subplots(2, 2, figsize=(12,8))

sns.violinplot(x = 'GTEP', y = 'TEY', data = turbine, scatter_kws={'alpha':0.6}, ax = axes[0,0])
sns.violinplot(x = 'TIT', y = 'TEY', data = turbine, scatter_kws={'alpha':0.6}, ax = axes[0,1])
sns.violinplot(x = 'TAT', y = 'TEY', data = turbine, scatter_kws={'alpha':0.6}, ax = axes[1,0])
sns.violinplot(x = 'CDP', y = 'TEY', data = turbine, scatter_kws={'alpha':0.6}, ax = axes[1,1])


# ## Neural Network :

# In[120]:


np.random.seed(8)

# split into input (X) and output (Y) variables
X = turbine.drop(['TEY'],axis=1)
Y = turbine['TEY']
X


# In[121]:


Y


# ### 1. Batch Size and Epochs

# In[136]:


# create model
model = Sequential()
model.add(Dense(26, input_dim = 10,  activation='relu')) #1st layer
model.add(Dense(26,  activation='tanh')) #2nd layer
model.add(Dense(26, activation='sigmoid')) #3rd layer 
model.add(Dense(26, activation='leaky_relu')) #4th layer or op layer


# In[137]:


# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_percentage_error'])


# In[138]:


# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=20, batch_size=10)


# In[139]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))


# In[140]:


model.metrics_names


# In[141]:


scores


# In[142]:


# Visualize training history

# list all data in history
history.history.keys()


# In[144]:


# summarize history for accuracy
plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('model mean_absolute_percentage_error')
plt.ylabel('mean_absolute_percentage_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[145]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # Hyperparameters all at once

# 
# The hyperparameter optimization was carried out by taking 2 hyperparameters at once. We may have missed the best values. The performance can be further improved by finding the optimum values of hyperparameters all at once given by the code snippet below.
# #### This process is computationally expensive.

# In[ ]:




