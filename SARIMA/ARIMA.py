a=0.1
x=[a,a*(1-a),a*(1-a)**2,a*(1-a)**3,a*(1-a)**4]
import matplotlib.pyplot as plt
plt.plot(x)
 
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from io import StringIO
import requests
import matplotlib.pyplot as plt
#%matplotlib inline

plt.style.use('bmh')

# Read the data

sales_data = pd.read_csv("C:/Users/parekhku/OneDrive - Merck Sharp & Dohme, Corp/Desktop/KP_Donotdelete/R/IMS/py/Time_Series_py/Tractor-sales.csv")
sales_data.head(5)
len(sales_data)

# since the complete date was not mentioned, we assume that it was the first of every month
dates = pd.date_range(start='2003-01-01', freq='MS',
                      periods=len(sales_data))

import calendar
sales_data['Month'] = dates.month
sales_data['Month'] = sales_data['Month'].apply(
        lambda x: calendar.month_abbr[x])
sales_data['Year'] = dates.year

sales_data.drop(['Month-Year'], axis=1, inplace=True)
sales_data.rename(columns={'Number of Tractor Sold':
    'Tractor-Sales'}, inplace=True)
sales_data = sales_data[['Month', 'Year', 'Tractor-Sales']]

# set the dates as the index of the dataframe, so that it can be treated as a time-series dataframe
sales_data.set_index(dates, inplace=True)
# check out first 5 samples of the data
sales_data.head(5)
sales_data.info()

# extract out the time-series
sales_ts = sales_data['Tractor-Sales']

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(sales_ts)
#plt.plot(np.log(sales_ts))
plt.xlabel('Years')
plt.ylabel('Tractor Sales')

#Determing rolling statistics
rolmean = sales_ts.rolling(window=12).mean()
rolstd = sales_ts.rolling(window=12).std()

#Plot rolling statistics:
orig = plt.plot(sales_ts, label='Original')
mean = plt.plot(rolmean, label='Rolling Mean')
std = plt.plot(rolstd, label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=True)

monthly_sales_data = pd.pivot_table(sales_data, 
                    values = "Tractor-Sales", 
                    columns = "Year", index = "Month")
monthly_sales_data = monthly_sales_data.reindex(index
        = ['Jan','Feb','Mar', 'Apr', 'May', 
           'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 
           'Nov', 'Dec'])
monthly_sales_data

# This is to see month on month plot by year. This will help us to understand if we have
# similar patterns in the time series
monthly_sales_data.plot()

# Making yearly data and ploting it to check year patterns
yearly_sales_data = pd.pivot_table(sales_data,
    values = "Tractor-Sales", columns = "Month", 
    index = "Year")
yearly_sales_data = yearly_sales_data[['Jan','Feb','Mar', 'Apr', 'May', 
                    'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                    'Nov', 'Dec']]
yearly_sales_data
yearly_sales_data.boxplot()

decomposition = sm.tsa.seasonal_decompose(sales_ts, 
                        model='additive')

fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of multiplicative time series')
plt.show()

plt.plot(np.log(sales_ts)) #Var stationary series
logseries=np.log(sales_ts)
meanstat=np.zeros((len(logseries)-1))
for i in range(len(logseries)-1):
    meanstat[i]=logseries[i+1]-logseries[i]
plt.plot(meanstat)#Mean stationary series
 
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

plot_acf(meanstat, lags=20) 
pyplot.show()
plot_pacf(meanstat, lags=20) 
pyplot.show()


# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
p = range(0, 4)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 
                 12) for x in
    list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


warnings.filterwarnings("ignore") # specify to ignore warning messages

AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(sales_ts,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue
        
        
print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))


# Let's fit this model
mod = sm.tsa.statespace.SARIMAX(sales_ts,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

pred1 = results.get_prediction(start='2003-01-01', dynamic=True)
pred1_ci = pred1.conf_int()

pred2 = results.get_forecast('2015-01-01')
pred2_ci = pred2.conf_int()

#In this case the model is used to predict data that the model was built on. 
#1-step ahead forecasting implies that each forecasted point is used to predict the 
#following one.
pred0 = results.get_prediction(start='2003-01-01', dynamic=False)
pred0_ci = pred0.conf_int()

#In sample prediction with dynamic forecasting of the last year  
#Again, the model is used to predict data that the model was built on.
pred1 = results.get_prediction(start='2003-01-01', dynamic=True)
pred1_ci = pred1.conf_int()

#"True" forecasting of out of sample data. 
#In this case the model is asked to predict data it has not seen before.
pred2 = results.get_forecast('2016-01-01')
# Give the end year till you want forecast
pred2_ci = pred2.conf_int()


#Plot the predicted values
ax = sales_ts.plot(figsize=(20, 16))
pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Monthly Tractor Sales')
plt.xlabel('Date')
plt.legend()
plt.show()

from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

plot_acf(sales_ts, lags=20) #MA 14
pyplot.show()

plot_pacf(sales_ts, lags=5)#AR 1
pyplot.show()

import numpy as np
import matplotlib.pyplot as plt
e=np.random.normal(loc=10, scale=10, size=500)
np.mean(e)
np.std(e)
x=10
rho=1
#from functools import reduce
#reduce((lambda x,y: 10*x+y), e)
plt.plot(x*rho+e)
