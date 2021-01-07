from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting.forecasting import plot_ys

y = load_airline()
y_train, y_test = temporal_train_test_split(y)
plot_ys(y_train, y_test, labels=["y_train", "y_test"])

'''
Before you create any sophisticated forecasts, it is helpful to compare your forecast to a naïve 
baseline — a good model must beat this value. sktime provides the NaiveForecaster method, with 
different “strategies”, to generate baseline forecasts.
The code and chart below demonstrate two naïve forecasts. 
The forecaster with strategy = “last” always predicts last observed value of the series. 
The forecaster with strategy = “seasonal_last” predicts the last value of the series 
observed in the given season. Seasonality in the example is specified as “sp=12”, or 12 months.
'''

from sktime.forecasting.naive import NaiveForecaster
import numpy as np
from sktime.performance_metrics.forecasting import smape_loss

naive_forecaster_last = NaiveForecaster(strategy="last")
naive_forecaster_last.fit(y_train)
fh = np.arange(len(y_test)) + 1
fh
y_last = naive_forecaster_last.predict(fh)


naive_forecaster_seasonal = NaiveForecaster(strategy="seasonal_last", sp=12)
naive_forecaster_seasonal.fit(y_train)
y_seasonal_last = naive_forecaster_seasonal.predict(fh)

plot_ys(y_train, y_test, y_last, y_seasonal_last, labels=["y_train", "y_test", "y_pred_last", "y_pred_seasonal_last"]);
smape_loss(y_last, y_test)

'''
The next forecast snippet shows how existing sklearn regressors can be easily and correctly 
adapted to forecasting tasks with minimal effort. Below, the sktime ReducedRegressionForecaster 
method forecasts the series using the the sklearnRandomForestRegressor model. Internally, sktime 
is splitting the training data into windows of length 12 for the regressor to train on.
'''
from sktime.forecasting.compose import ReducedRegressionForecaster
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss

regressor = RandomForestRegressor()
forecaster = ReducedRegressionForecaster(regressor, window_length=12)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

plot_ys(y_train, y_test, y_pred, labels=['y_train', 'y_test', 'y_pred'])
smape_loss(y_test, y_pred)

'''
sktime also contains native forecasting methods, such as AutoArima.
'''
#pip install pmdarima --user
import pmdarima
from sktime.forecasting.arima import AutoARIMA
forecaster = AutoARIMA(sp=12)
forecaster.fit(y_train)

y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
smape_loss(y_test, y_pred)


'''
sktime has a number of statistical forecasting algorithms, based on 
implementations in statsmodels. For example, to use exponential smoothing with an 
additive trend component and multiplicative seasonality, we can write the following.

Note that since this is monthly data, the seasonal periodicity (sp), or the number of 
periods per year, is 12.
'''

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
forecaster = ExponentialSmoothing(trend="add", seasonal="multiplicative", sp=12)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
smape_loss(y_test, y_pred)


'''
Compositite model building
sktime provides a modular API for composite model building for forecasting.

Ensembling
Like scikit-learn, sktime provides a meta-forecaster to ensemble multiple forecasting algorithms. 
For example, we can combine different variants of exponential smoothing as follows:
'''
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import EnsembleForecaster
forecaster = EnsembleForecaster([
    ("ses", ExponentialSmoothing(seasonal="multiplicative", sp=12)),
    ("holt", ExponentialSmoothing(trend="add", damped=False, seasonal="multiplicative", sp=12)),
    ("damped", ExponentialSmoothing(trend="add", damped=True, seasonal="multiplicative", sp=12))
])
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
smape_loss(y_test, y_pred)