import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from data_prep import make_lags
from sklearn.model_selection import train_test_split
from my_reservoir import my_reservoir
from eval import rmse

random.seed(10)
np.random.seed(10)

##### ----- observe airplane data ----- #####
orig_data = load_airline()

# # basics
# print(orig_data.shape)
# print(orig_data.head())
# orig_data.plot()
# plt.show()

# # autocorrelation
# plot_acf(orig_data)
# plot_pacf(orig_data, lags=30)
# plt.show()

##### ----- data preparation ----- #####
num_lags = 12
# consider adding seasonality    
    
df = orig_data.to_frame()
df = df.rename(columns={'Number of airline passengers': 'num_passengers'})

# add time steps
df['time_steps'] = np.arange(len(df.index))

# create lags
make_lags(df, num_lags)
df = df.dropna()

# divide df into features and target
u = df.iloc[:, df.columns != 'num_passengers']
y = df.iloc[:, 0]

# divide X,y into train and test set
u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.2, shuffle=False)


##### ----- train data ----- #####
alpha = 0.01
# should consider adding washout; limited by small data size

# set up reservoir and get readout
reservoir = my_reservoir(n_inputs=u_train.shape[1], n_neurons=30, rhow=1.25, leak_range=(0.1, 0.3))
X_train = reservoir.forward(u_train, collect_states=True)

# train readout with ridge regression
R = X_train.T @ X_train
P = X_train.T @ y_train
wout = np.linalg.inv((R + alpha * np.eye(X_train.shape[1]))) @ P
y_pred_train = X_train @ wout


##### ----- evaluate performance and test ----- #####
print(f'Training accuracy: {rmse(y_train.to_numpy(), y_pred_train)}')
# probably try other accuracy measurements besides rmse

X_test = reservoir.forward(u_test, collect_states=True)
y_pred_test = X_test @ wout

print(f'Testing accuracy: {rmse(y_test.to_numpy(), y_pred_test)}')

t = np.arange(y_train.shape[0])
t_new = np.arange(y_train.shape[0] +1, y_train.shape[0] + 1 + y_test.shape[0])

plt.plot(t, y_train.to_numpy())
plt.plot(t, y_pred_train)

plt.plot(t_new, y_test.to_numpy())
plt.plot(t_new, y_pred_test)
plt.show()