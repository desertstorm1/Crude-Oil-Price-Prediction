#!/usr/bin/env python
# coding: utf-8

# # CQF FINAL PROJECT MODULE 6 2018 
#                  DR Abdulaziz Alheraiqi

# given the time-contraint I m going to focus as a first step to build a predicting mode given just data of past crude oil price , gold,SP500,down jones index and USD index . For this we retrive The data  from QUNADL. We could have consider other data sets like the Opec basket or Brent but we are considering WTI leading the direction Of crude oil prices.
# combining the threee data sets(Gold,DJ index,USD Index ) to predict WTI crude oils prices 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #               ## importing Data

# In[2]:


wti= pd.read_csv("wti.csv", parse_dates=['Date'])


# In[3]:


gold=pd.read_csv("gold.csv", parse_dates=['Date'])


# In[4]:


usdolex=pd.read_csv("FRED-USDollarIndex.csv", parse_dates=['Date'])


# In[5]:


snp500=pd.read_csv("snp500.csv", parse_dates=['Date'])


# In[6]:


dji=pd.read_csv("dji.csv", parse_dates=['Date'])


# #checking the structure of the dataset

# In[7]:


wti.head(5)


# In[8]:


gold.head(5)


# In[9]:


usdolex.head(5)


# In[10]:


snp500.head(5)


# In[11]:


dji.head(5)


# #Number of rows and columns

# In[12]:


wti.shape


# In[13]:


gold.columns


# In[14]:


snp500.columns


# In[15]:


dji.columns


# #name of columns

# In[16]:


wti.columns
gold.columns
usdolex.columns
snp500.columns
dji.columns


# #Check the Datatypes

# In[17]:


wti.dtypes
gold.dtypes
usdolex.dtypes
snp500.dtypes
dji.dtypes


# # Visualization

# In[18]:


wti.hist()
plt.show()


# In[19]:


gold.hist()
plt.show()


# In[20]:


snp500.hist()
plt.show()


# In[21]:


dji.hist()
plt.show()


# # #Check for Missing Values

# In[22]:


wti.isnull().any()


# In[23]:


gold.isnull().any()


# In[24]:


usdolex.isnull().any()


# In[25]:


snp500.isnull().any()


# In[26]:


dji.isnull().any()


# # No missing values Good Data

# #Now try to merge and see if the data is okay

# In[27]:


wti.set_index('Date', inplace=True)


# In[28]:


gold.set_index('Date', inplace=True)


# In[29]:


usdolex.set_index('Date', inplace=True)


# In[30]:


snp500.set_index('Date', inplace=True)


# In[31]:


dji.set_index('Date', inplace=True)


# # Merge the Datasets

# In[32]:


#first merge wti and gold then i add USD to the merged and I
#add SNP500 to the merged and finaly i add dji.
#using inner joint.


# In[33]:


merged=pd.merge(wti, gold, how='inner', on='Date')


# In[34]:


merged=pd.merge(merged, usdolex, how='inner', on='Date')


# In[35]:


merged=pd.merge(merged, snp500, how='inner', on='Date')


# In[36]:


merged=pd.merge(merged, dji, how='inner', on='Date')


# In[37]:



merged.head()


# In[38]:


mergedalldata = merged


# In[39]:


merged.shape


# In[40]:


merged.columns


# # Rename the columns

# In[41]:


merged.columns = ['WTIPrice','GoldPrice','USDOLEX','snpClose','djiClose']


# In[42]:


merged.head()


# In[43]:


merged['WTIPrice'].plot()
plt.show()


# In[44]:


merged['WTIPrice'].max()


# # Lehman Brothers
# #03/07/2008	145.31
# 

# #Check out the dtypes

# In[45]:


merged.dtypes


# # DATA PREPROCESSING

# #Check the null values

# In[46]:


merged.isnull().any().sum()


# In[47]:


merged=merged.dropna()


#  #Check on the Outliers and remove them from our observations

# #Visualizing

# In[48]:


merged.hist()
plt.show()


# In[49]:


#Calculating Daily Return  with two methods


# In[50]:


daily_PC = merged/merged.shift(1)-1


# In[51]:


daily_PC .head()


# In[52]:


daily_PC.corr()


# In[53]:


Pct = merged.pct_change()


# In[54]:


Pct.head(5)


# In[55]:


Pct.hist()
plt.show()


# #The Daily Return columns are not giving a significant correlation result and will not be 
# #used in prediction

# In[56]:


# EDA Statistics


# In[57]:


merged.describe()


# In[58]:


merged.median()


# In[59]:


Pct.describe()


# In[60]:


Pct.median()


# In[61]:



#Five Point Summary


# In[62]:


merged.min()


# In[63]:


merged.min()


# In[64]:


merged.quantile()


# In[65]:


merged.quantile(0.25)


# In[66]:


merged.quantile(0.75)


# # Shape of the curve

# In[67]:


merged.kurtosis()


# In[68]:


merged.skew()


# 
# #Checking for Outliers
# #Location of the outliers

# In[69]:


merged.boxplot()
plt.show()


# In[70]:


merged.boxplot(column='WTIPrice', return_type='axes')
plt.show()


# In[71]:


merged.boxplot(column='GoldPrice', return_type='axes')
plt.show()


# In[72]:


merged.boxplot(column='GoldPrice', return_type='axes')
plt.show()


# In[73]:


merged.boxplot(column='snpClose', return_type='axes')
plt.show()


# In[74]:


merged.boxplot(column='djiClose', return_type='axes')
plt.show()


# # Data Outliers

# In[75]:


merged.quantile()


# In[76]:


merged.quantile(0.25) * 1.5


# In[77]:


merged.quantile(0.75) * 1.5


# In[78]:


wtiQ1=merged['WTIPrice'].quantile(0.25) * 1.5


# In[79]:


print(wtiQ1)


# In[80]:


wtiQ3=merged['WTIPrice'].quantile(0.75) *1.5
print(wtiQ3)


# In[81]:


goldQ1=merged['GoldPrice'].quantile(0.25) *1.5
print(goldQ1)


# In[82]:


goldQ3=merged['GoldPrice'].quantile(0.75) * 1.5


# In[83]:


usdolexQ1=merged['USDOLEX'].quantile(0.25) *1.5
print(usdolexQ1)


# In[84]:


usdolexQ3=merged['USDOLEX'].quantile(0.75) *1.5
print(usdolexQ3)


# In[85]:


snp500Q1=merged['snpClose'].quantile(0.25) * 1.5
print(snp500Q1)


# In[86]:


snp500Q1=merged['snpClose'].quantile(0.75) * 1.5


# In[87]:


snp500Q3=merged['snpClose'].quantile(0.75) * 1.5


# In[88]:


djiQ1=merged['djiClose'].quantile(0.25) * 1.5
print(djiQ1)


# In[89]:


djiQ3=merged['djiClose'].quantile(0.75) * 1.5
print(djiQ3)


# In[90]:


#Counting the number of outliers


# In[91]:


merged.head(5)


# In[92]:


merged['WTIPrice'].loc[merged['WTIPrice'] <=wtiQ1].count()


# In[93]:


merged['WTIPrice'].loc[merged['WTIPrice'] >=wtiQ3].count()


# In[94]:


merged['GoldPrice'].loc[merged['GoldPrice'] <=goldQ1].count()


# In[95]:


merged['GoldPrice'].loc[merged['GoldPrice'] >=goldQ3].count()


# In[96]:


merged['USDOLEX'].loc[merged['USDOLEX'] <=usdolexQ1].count()


# In[97]:


merged['USDOLEX'].loc[merged['USDOLEX'] >=usdolexQ3].count()


# In[98]:


merged['snpClose'].loc[merged['snpClose'] <=snp500Q1].count()


# In[99]:


merged['snpClose'].loc[merged['snpClose'] >=snp500Q3].count()


# In[100]:


merged['djiClose'].loc[merged['djiClose'] <=djiQ1].count()


# In[101]:


merged['djiClose'].loc[merged['djiClose'] >=djiQ3].count()


# #Decide on Outlier treatment
# #as outlier constitute significant part of data
# #the decision is to keep the outlier

# #Is there a relationship between Gold and Oil

# In[102]:


merged.corr()


# In[ ]:





# In[103]:


## there stronge positive corrrelation between Crude oil prices and Gold 
## prices (.8)and negative correlation with USD(-.75)
## and positive correlation with stock market more with adj Dow jones close


# In[104]:


#Decision to drop Snp500 and keep DJI as both represent the stock market and having both of them puts
#too much of a weight on stock market for our prediction


# In[105]:


merged.head(5)


# In[106]:


merged=merged.drop('snpClose', axis=1)


# In[107]:


merged.columns


# In[108]:


merged.head(5)


# # Prediction

# In[109]:


#The Prediction works only in standard circumstances barring acts of 
#God such as
#War and Natural Calamities
#The assumption is that there exists a Linear relationship between 
#Features and Target V


# In[ ]:





# # ##Tomorrow's price Prediction 
# ###based on Regression

# In[110]:


import sklearn


# In[111]:


from sklearn.model_selection import cross_val_score as cross_validation


# In[112]:


from sklearn.utils import shuffle


# In[113]:


from sklearn import linear_model


# In[114]:


from sklearn.tree import DecisionTreeRegressor


# In[115]:


from sklearn.ensemble import RandomForestRegressor


# In[116]:


from sklearn.metrics import mean_squared_error, r2_score


# In[117]:


from sklearn.model_selection import train_test_split


# In[118]:


#Splitting data into Target variable and features


# In[119]:


#targetprice=merged['WTIPrice']Splitting data into Target variable 
#and features


# In[120]:


targetprice= merged['WTIPrice']


# In[121]:


targetprice.head()


# In[122]:


features=merged[['GoldPrice','USDOLEX','djiClose']]
features.head(5)


# In[123]:


features.columns


# In[124]:


#Shuffling the data


# In[125]:



features=shuffle(features,random_state=0)


# In[126]:


targetprice=shuffle(targetprice, random_state=0)


# In[127]:


targetprice.head(5)


# In[128]:


features.head(5)


# # Dividing data into Train and Test

# In[129]:


from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[130]:


X_train, X_test, y_train, y_test = train_test_split(
  features,targetprice , test_size=0.4, random_state=0)


# In[131]:


X_train.head(5)


# In[132]:


y_train.head(5)


# In[133]:


y_train.describe()


# In[134]:


X_train.describe()


# In[135]:


X_test.describe()


# In[136]:


y_test.describe()


# # Linear Regression on WTI and Gold data

# In[137]:


regr_1= linear_model.LinearRegression()


# In[138]:


regr_1.fit(X_train, y_train)


# # #Decision Tree Regressor

# In[139]:


regr_2 = DecisionTreeRegressor(max_depth=2)


# In[140]:


regr_2.fit(X_train, y_train)


# # Random Forest Regressor

# In[141]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[142]:


rf.fit(X_train, y_train)


# # Making the prediction from Test data

# In[143]:


y_pred= regr_1.predict(X_test)


# In[144]:


y_pred= regr_2.predict(X_test)


# In[145]:


y_pred= rf.predict(X_test)


# In[146]:


#Metrics for Linear Regression only


# In[147]:


print("Coefficients: ", regr_1.coef_)


# In[148]:


#Mean squared error


# In[149]:


print("mean squared error:  ",mean_squared_error(y_test,y_pred))


# In[150]:


#Variance score
print("Variance score: ",   r2_score(y_test, y_pred))


# In[151]:


#Accuracy Score for other algorithms than Linear Regression


# In[152]:


regr_1.score(X_test, y_test)


# In[153]:


regr_2.score(X_test, y_test)


# In[154]:


rf.score(X_test, y_test)


# In[155]:


#Recommendation is to use Decision Tree Regressor 
#Random Forest Regressor is overfitting in this scenario


# In[156]:


#STANDARD DEVIATION


# In[157]:


stdprc=targetprice.std()
stdprc


# In[158]:


merged.head(5)


# In[171]:


gold.head(5)


# In[172]:


WTINextDayPredict=regr_1.predict([[1257.6,95.25,19000]])


# In[173]:


WTINextDayPredict


# In[174]:


WTINextDayPredict=regr_2.predict([[1257.6,95.25,19000]])


# In[175]:


WTINextDayPredict


# In[176]:


WTINextDayPredict=rf.predict([[1257.6,95.25,19000]])


# In[177]:


WTINextDayPredict


# In[178]:


#_x_ is the standard deviation of the Diff between Open 
#and Close of sensex so this rang


# In[179]:


print("WTI Next Day Price Prediction Likely to Close at: ",WTINextDayPredict , "(+-Standard Deviation)")


# In[180]:


print("WTI Next Day Price: ",WTINextDayPredict+stdprc , " & " , WTINextDayPredict-stdprc)


# # Finding how far I need to take historical data to improve my prediction

# In[183]:


##doing the same prediction for 10 years,7 years,5 years,3years,2years and 
##one year
##I ploted the correlation and the prediction accuracy in 
##seperate excel sheet attached (Model_Evaluation_Analysis)


# In[ ]:





# # Autoregressive Integrated Moving Average Model

# In[184]:


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 


# In[185]:


from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')


# In[186]:


series = wti.head(5000)


# In[187]:


series.describe()


# In[188]:


autocorrelation_plot(series)
pyplot.show()


# # fit model

# In[189]:


model = ARIMA(series, order=(5,1,0))


# In[190]:


model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[191]:


# plot residual errors


# In[192]:


residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()


# In[193]:


residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# The distribution of the residual errors is displayed.
# The results show that indeed there is No  bias in the prediction (a zero mean in the residuals).

# Rolling Forecast ARIMA Model

# In[194]:


from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 


# In[195]:


X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()


# In[196]:


for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


# In[198]:


error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# In[199]:


# plot


# In[200]:


from matplotlib import pyplot


# In[202]:


pyplot.plot(test)
pyplot.show()


# In[203]:


pyplot.plot(predictions, color='red')
pyplot.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




