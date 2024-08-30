#!/usr/bin/env python
# coding: utf-8

# # Forecasting Coka Cola Assignment

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


# ## 2. Coka Cola Dataset

# In[3]:


coke=pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\CocaCola_Sales_Rawdata.xlsx",
                    index_col=0, 
                    parse_dates=True)
coke


# In[4]:


coke.info()


# ## Visualizations

# In[5]:


coke = pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\CocaCola_Sales_Rawdata.xlsx",
                     index_col = 0,
                     header = 0,
                     parse_dates = True)
coke


# In[6]:


coke.index


# In[7]:


plt.figure(figsize = (15,7))
plt.plot(coke)


# #### Histogram and Density Plots

# In[8]:


# create a density plot
coke.plot(kind='kde')


# In[9]:


# create a histogram plot
coke.hist()


# #### Lag plot

# In[10]:


# create a scatter plot
plt.figure(figsize = (17,7))
pd.plotting.lag_plot(coke)


# In[11]:


# create an autocorrelation plot
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize = (50,15))
plot_acf(coke, lags=6)
plt.show()


# #### Box and Whisker Plots by Interval

# In[13]:


coke = pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\CocaCola_Sales_Rawdata.xlsx",
                     index_col = 0,
                     header = 0,
                     parse_dates = True,
                    squeeze=True)
coke


# In[14]:


type(coke)


# ### Sampling and Basic Transformations 

# In[15]:


coke = pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\CocaCola_Sales_Rawdata.xlsx",
                     index_col = 0,
                     header = 0,
                     parse_dates = True,
                    squeeze=True)
coke


# In[16]:


coke.shape


# In[18]:


coke = pd.read_excel(r"C:\Users\gupta\Downloads\Data Science Assesment\Forecasting\CocaCola_Sales_Rawdata.xlsx")


# In[19]:


quarter =['Q1','Q2','Q3','Q4']


# In[20]:


p = coke["Quarter"][0]
p[0:2]
coke['quarter']= 0

for i in range(42):
    p = coke["Quarter"][i]
    coke['quarter'][i]= p[0:2]

coke


# In[21]:


quarter_dummies = pd.DataFrame(pd.get_dummies(coke['quarter']))
quarter_dummies


# In[22]:


coke=pd.concat([coke,quarter_dummies],axis=1)
coke


# In[23]:


# Complete the dataset
coke['t']=np.arange(1,43)
coke['t_square']=np.square(coke.t)
coke['log_Sales']=np.log(coke.Sales)
coke


# # Forecasting - Model Based Methods 

# In[24]:


coke


# In[25]:


coke['Sales'].plot()


# In[26]:


plt.figure(figsize=(12,4))
sns.lineplot(x="quarter",y="Sales",data=coke)


# ## Splitting data

# In[27]:


coke


# In[28]:


# For self understanding of forecasting values data split into multiples of 12
Train = coke.head(25)
Test = coke.tail(7)


# In[29]:


Train


# In[30]:


Test


# In[31]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[32]:


#Exponential
Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[33]:


#Quadratic 
Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# In[34]:


#Additive seasonality 
add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1', 'Q2', 'Q3', 'Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[35]:


#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1', 'Q2', 'Q3', 'Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[36]:


##Multiplicative Seasonality
Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[37]:


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


# In[38]:


#Multiplicative Seasonality Quadratic 
mult_sea_Quad = smf.ols('log_Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_mult_sea_quad = pd.Series(mult_sea_Quad.predict(Test[['Q1', 'Q2', 'Q3', 'Q4','t','t_square']]))
rmse_mult_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_mult_sea_quad))**2))
rmse_mult_sea_quad


# In[39]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[40]:


#Build the model on entire data set
model_full = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=coke).fit()


# In[41]:


pred_new  = pd.Series(model_full.predict(coke))
pred_new


# In[42]:


coke["forecasted_Sales"] = pd.Series(np.exp(pred_new))


# In[43]:


plt.figure(figsize=(18,10))
plt.plot(coke[['Sales','forecasted_Sales']].reset_index(drop=True))


# ### Splitting data

# In[44]:


Train = coke.head(35)
Test = coke.tail(7)


# # Moving Average 

# In[45]:


plt.figure(figsize=(24,7))
coke['Sales'].plot(label="org")
coke["Sales"].rolling(4).mean().plot(label=str(5))
plt.legend(loc='best')


# In[46]:


plt.figure(figsize=(24,7))
coke['Sales'].plot(label="org")
for i in range(2,18,6):
    coke["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


# # Time series decomposition plot 

# In[47]:


decompose_ts_add = seasonal_decompose(coke['Sales'], period = 12)
decompose_ts_add.plot()
plt.show()


# ## ACF plots and PACF plots

# In[48]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(coke.Sales,lags=12)
tsa_plots.plot_pacf(coke.Sales,lags=12)
plt.show()


# ### Evaluation Metric MAPE

# In[49]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# ### Simple Exponential Method

# In[50]:


ses_model = SimpleExpSmoothing(Train["Sales"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) 


# ### Holt method 

# In[51]:


# Holt method 
hw_model = Holt(Train["Sales"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) 


# ### Holts winter exponential smoothing with additive seasonality and additive trend

# In[52]:


hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)


# ### Holts winter exponential smoothing with multiplicative seasonality and additive trend

# In[53]:


hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)


# ## Final Model by combining train and test

# In[54]:


hwe_model_mul_add = ExponentialSmoothing(coke["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit() 


# In[55]:


#Forecasting for next 7 time periods
hwe_model_mul_add.forecast(7)


# #### Inference: Holts Winter ES with Multilicative Seasonal & additive gives minimum error 2.52

# In[ ]:




