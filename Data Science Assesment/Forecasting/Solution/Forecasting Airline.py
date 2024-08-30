#!/usr/bin/env python
# coding: utf-8

# # Forecasting Assignment

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

import itertools
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


# ## 1. Airlines Dataset

# In[2]:


air = pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\Airlines+Data.xlsx",
                       index_col=0, 
                       parse_dates=['Month'])
air


# ## Visualizations

# In[3]:


air.info()


# In[4]:


air.index


# In[5]:


plt.figure(figsize = (15,7))
plt.plot(air)


# In[9]:


air = pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\Airlines+Data.xlsx",
                       index_col=0, 
                       header=0,
                       parse_dates=True)
air


# #### Histogram and Density Plots

# In[10]:


# create a histogram plot
air.hist()


# In[11]:


# create a density plot
air.plot(kind='kde')


# In[12]:


air = pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\Airlines+Data.xlsx",
                       index_col=0, 
                       header=0,
                       parse_dates=True,
                       squeeze=True)
air


# In[13]:


type(air)


# In[14]:


# Grouping by Year
groups = air.groupby(pd.Grouper(freq='A'))
groups


# In[15]:


years = pd.DataFrame()

for name, group in groups:
    years[name.year] = group.values

years


# In[16]:


plt.figure(figsize = (15,7))
years.boxplot()


# #### Lag plot

# In[18]:


# create a scatter plot
plt.figure(figsize = (15,9))
pd.plotting.lag_plot(air)


# In[19]:


# create an autocorrelation plot
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize = (32,20))
plot_acf(air, lags=95)
plt.show()


# ### Sampling and Basic Transformations 

# #### Upsampling Data

# In[20]:


air = pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\Airlines+Data.xlsx",
                       index_col=0, 
                       header=0,
                       parse_dates=True,
                       squeeze=True)
air


# In[22]:


air.shape


# In[23]:


upsampled = air.resample('D').mean()
upsampled.head(20)


# In[24]:


upsampled.shape


# ##### Interpolate the Missing Value

# In[25]:


interpolated = upsampled.interpolate(method='linear')
interpolated.head(30)


# In[26]:


interpolated.plot()


# In[28]:


air.plot()


# #### Downsampling Data

# In[29]:


# downsample to quarterly intervals
resample = air.resample('Q')
quarterly_mean_sales = resample.mean()


# In[30]:


quarterly_mean_sales.plot()


# ## Tranformations

# In[31]:


# load and plot a time series
c,index_col=0,header=0, parse_dates=True)
air


# In[33]:


# line plot
plt.subplot(211)
plt.plot(air)

# histogram
plt.subplot(212)
plt.hist(air)

plt.show()


# #### Log Transform

# In[34]:


dataframe = pd.DataFrame(np.log(air.values), columns = ['Passengers'])
dataframe


# In[35]:


# line plot
plt.subplot(211)
plt.plot(dataframe['Passengers'])

# histogram
plt.subplot(212)
plt.hist(dataframe['Passengers'])
plt.show()


# In[36]:


quarterly_mean_sales.head()


# #### Square Root Transform

# In[37]:


dataframe = pd.DataFrame(np.sqrt(air.values), columns = ['Passengers'])
dataframe


# In[38]:


# line plot
plt.subplot(211)
plt.plot(dataframe['Passengers'])

# histogram
plt.subplot(212)
plt.hist(dataframe['Passengers'])
plt.show()


# # Forecasting - Model Based Methods 

# In[39]:


air=pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\Airlines+Data.xlsx")
air


# In[40]:


air['Passengers'].plot()


# In[41]:


air["month"] = air['Month'].dt.strftime("%b") # month extraction
air["year"] = air['Month'].dt.strftime("%Y") # year extraction


# In[42]:


air


# In[43]:


mp = pd.pivot_table(data = air,
                                 values = "Passengers",
                                 index = "year",
                                 columns = "month",
                                 aggfunc = "mean",
                                 fill_value=0)
mp


# In[44]:


plt.figure(figsize=(12,8))
sns.heatmap(mp,
            annot=True,
            fmt="g",
            cmap = 'YlGnBu') #fmt is format of the grid values


# In[45]:


# Boxplot for ever
plt.figure(figsize=(15,10))

plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=air)

plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=air)


# In[47]:


plt.figure(figsize=(17,8))
sns.lineplot(x="year",y="Passengers",data=air)


# ## Splitting data

# In[48]:


air


# In[41]:


air.shape


# In[49]:


# Complete the dataset
air['t']=np.arange(1,97)
air['t_square']=np.square(air.t)
air['log_Passengers']=np.log(air.Passengers)
air2=pd.get_dummies(air['month'])


# In[50]:


air


# In[51]:


air2


# In[52]:


air=pd.concat([air,air2],axis=1)
air


# In[54]:


# For self understanding of forecasting values data split into multiples of 12
Train = air.head(84)
Test = air.tail(12)


# In[55]:


Train


# In[56]:


Test


# In[66]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


# In[67]:


#Exponential
Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[68]:


#Quadratic 
Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


# In[69]:


#Additive seasonality 
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[70]:


#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[71]:


##Multiplicative Seasonality
Mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[72]:


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


# In[73]:


#Multiplicative Seasonality Quadratic 
mult_sea_Quad = smf.ols('log_Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_mult_sea_quad = pd.Series(mult_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_mult_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_mult_sea_quad))**2))
rmse_mult_sea_quad


# In[74]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[78]:


#Build the model on entire data set
model_full = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=air).fit()


# In[79]:


pred_new  = pd.Series(model_full.predict(air))
pred_new


# In[80]:


air["forecasted_Passengers"] = pd.Series(np.exp(pred_new))


# In[81]:


plt.figure(figsize=(15,10))
plt.plot(air[['Passengers','forecasted_Passengers']].reset_index(drop=True))


# ### Splitting data

# In[86]:


air=pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\Airlines+Data.xlsx")
Train = air.head(84)
Test = air.tail(12)


# In[87]:


Train


# In[88]:


Test


# # Moving Average 

# In[92]:


plt.figure(figsize=(24,7))
air['Passengers'].plot(label="org")
air["Passengers"].rolling(15).mean().plot(label=str(5))
plt.legend(loc='best')


# In[93]:


plt.figure(figsize=(24,7))
air['Passengers'].plot(label="org")
for i in range(2,24,6):
    air["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


# # Time series decomposition plot 

# In[94]:


decompose_ts_add = seasonal_decompose(air['Passengers'], period = 12)
decompose_ts_add.plot()
plt.show()


# ## ACF plots and PACF plots

# In[95]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(air.Passengers,lags=12)
tsa_plots.plot_pacf(air.Passengers,lags=12)
plt.show()


# ### Evaluation Metric MAPE

# In[96]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# ### Simple Exponential Method

# In[97]:


ses_model = SimpleExpSmoothing(Train["Passengers"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)


# ### Holt method 

# In[98]:


# Holt method 
hw_model = Holt(Train["Passengers"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers) 


# ### Holts winter exponential smoothing with additive seasonality and additive trend

# In[99]:


hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers) 


# ### Holts winter exponential smoothing with multiplicative seasonality and additive trend

# In[100]:


hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)


# ## Final Model by combining train and test

# In[102]:


hwe_model_mul_add = ExponentialSmoothing(air["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit() 


# In[103]:


#Forecasting for next 12 time periods
hwe_model_mul_add.forecast(12)


# # Forecasting using Auto ARIMA model 

# In[106]:


import statsmodels.tsa.seasonal
#!pip install pmdarima
from pmdarima import auto_arima


# In[107]:


air=pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\Airlines+Data.xlsx",header=0, index_col=0, parse_dates=True)
air


# In[108]:


air.plot()


# In[109]:


from pmdarima.arima import ADFTest
adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(air)


# In[110]:


train = air[:85]
test = air[-20:]
plt.plot(train)
plt.plot(test)


# In[111]:


arima_model = auto_arima(train, start_p=0, d=1, start_q=0,
                        max_p=5, max_d=5, max_q=5, start_P=0,
                        D=1, start_Q=0, max_P=5, max_D=5,
                        max_Q=5, m=12, seasonal=True,
                        error_action='warn',trace=True,
                        suppress_warnings=True,stepwise=True,
                         random_state=20,n_fits=50)


# In[112]:


arima_model.summary()


# In[113]:


prediction = pd.DataFrame(arima_model.predict(n_periods = 20),index=test.index)
prediction.columns = ['predicted_pass']
prediction


# In[114]:


plt.figure(figsize=(8,5))
plt.plot(train,label='Training')
plt.plot(test, label='Test')
plt.plot(prediction, label='Predicted')
plt.legend(loc = 'upper left')
plt.show()


# #### Persistence/ Base model

# In[115]:


# evaluate a persistence model and load data
train=pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\Airlines+Data.xlsx", header=None, index_col=0, parse_dates=True, squeeze=True)
train


# In[116]:


# prepare data
X = train.values
X


# In[ ]:





# In[118]:


train_size = int(len(X) * 0.50)
train_size


# In[119]:


train, test = X[0:train_size], X[train_size:]


# In[120]:


train


# In[121]:


test


# In[122]:


# walk-forward validation
history = [x for x in train]
import warnings
history


# ## Predictions

# In[123]:


predictions = list()

for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
    
    # observation
    obs = test[i]
    history.append(obs)
    
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))


# In[126]:


# report performance
rmse = np.sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


# #### Inference: By using Auto ARIMA Model we get values for ARIMA as (0,1,1)(1,1,0) [12]. 

# In[ ]:





# In[ ]:




