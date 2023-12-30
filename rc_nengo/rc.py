import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nengo
from sktime.datasets import load_airline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def make_lags(df, num_lags):
    '''
        Function used to add lags to pandas dataframed

        df: dataframe to add lags to; column you want to lag must be at col index 0
        num_lags: number of lag features to add to dataframe
    '''
    for i in range(1, num_lags+1):
        lag_name = f'lag_{i}'
        df[lag_name] = df.iloc[:, 0].shift(i)


##### ----- get airplane data ----- #####
orig_data = load_airline()

##### ----- data preparation ----- #####
num_lags = 12

df = orig_data.to_frame()
df = df.rename(columns={'Number of airline passengers': 'num_passengers'})

# add time steps
df['time_steps'] = np.arange(len(df.index))

# add month
df['month'] = df.index.month

# add year
df['year'] = df.index.year

# create lags
make_lags(df, num_lags)
df = df.dropna()

# print(df)

# divide df into features and target
u = df.iloc[:, df.columns != 'num_passengers']
y = df.iloc[:, 0]

# divide u,y into train and test set
# u_train, u_test, y_train, y_test = train_test_split(u, y, test_size=0.25, shuffle=False)

# print(u_train.shape)
# print(u_test.shape)
# print(y_train.shape)
# print(y_test.shape)


##### ----- implement reservoir using nengo ----- #####
# task parameters
pulse_interval = 1.0
amplitude = 0.1
# freq = 3.0
# decay = 2.0
dt = 0.002
trials_train = 3
trials_test = 2

# fixed model params
n = 200
seed = 0
rng = np.random.RandomState(seed)
ens_kwargs = dict(  # neuron parameters
    n_neurons=n,
    dimensions=1,
    neuron_type=nengo.LIF(),  # nengolib.neurons.Tanh()
    intercepts=[-1]*n,  # intercepts are irelevant for Tanh
    seed=seed,
)

# Hyper-parameters
tau = 0.1                   # lowpass time-constant (10ms in [1])
# tau_learn = 0.1             # filter for error / learning (needed for spiking)
tau_probe = 0.05            # filter for readout (needed for spiking)
# learning_rate = 0.1         # 1 in [1]
g = 1.5 / 400               # 1.5 in [1], scaled by firing rates
g_in = tau / amplitude      # scale the input encoders (usually 1)
# g_out = 1.0                 # scale the recurrent encoders (usually 1)

# Pre-computed constants
T_train = trials_train * pulse_interval
T_total = (trials_train + trials_test) * pulse_interval

# set up input and output (different from tutorial)
with nengo.Network(seed=seed) as model:
    in_u = nengo.Node(output=nengo.processes.PresentInput(u, dt))
    z_out = nengo.Node(output=nengo.processes.PresentInput(y, dt))
    
# Initial weights
e_in = g_in * rng.uniform(-1, +1, (n, u.shape[1]))  # fixed encoders for f_in (u_in)
# e_out = g_out * rng.uniform(-1, +1, (n, 1))  # fixed encoders for f_out (u)
res_weights = rng.randn(n, n) * g / np.sqrt(n)  # target-generating weights (variance g^2/n) (weights for reservoir)

# model
with model:
    xD = nengo.Ensemble(**ens_kwargs) # reservoir
    sD = nengo.Node(size_in=n)  # pre filter
    
    nengo.Connection(in_u, sD, synapse=None, transform=e_in) # filter input
    nengo.Connection(sD, xD.neurons, synapse=tau) # filtered input to reservoir
    nengo.Connection(xD.neurons, sD, synapse=None, transform=res_weights)  # chaos
    
# probes
with model:
    p_readout = nengo.Probe(xD.neurons, synapse=tau_probe)
    p_target = nengo.Probe(z_out, synapse=tau_probe)
    
# run simulation
with nengo.Simulator(model, dt=dt) as sim:
    sim.run(T_total)
    
# time steps of sim for training and testing
t_train = sim.trange() < T_train
t_test = sim.trange() >= T_train

print("IMPORTANT")
print(sim.data[p_target])
print(sim.data[p_target].shape)


##### ----- train readout ----- #####
train_target = sim.data[p_target][t_train]

# train readout weights with least-squares L2 regularization Nengo solver
# solver = nengo.solvers.LstsqL2(reg=1e-2)
# w_star, _ = solver(sim.data[p_readout][t_train], train_target)
# pred = sim.data[p_readout].dot(w_star)
# train_pred = pred[t_train]

# train readout with ridge regression
rr_model = Ridge()
rr_model.fit(sim.data[p_readout][t_train], train_target)
train_pred = sim.data[p_readout][t_train].dot(rr_model.coef_.T)
# train_pred = pred[t_train]

# mse
mse_train = np.mean((train_target - train_pred) ** 2)
print("Training Mean Squared Error:", mse_train)

# graph training results
plt.plot(sim.trange()[t_train], train_pred, label="prediction")
plt.plot(sim.trange()[t_train], train_target, label="target")
plt.title("Training Output")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
plt.show()


##### ----- evaluation ----- #####
test_pred = sim.data[p_readout][t_test].dot(rr_model.coef_.T)
test_target = sim.data[p_target][t_test]

# mse
mse_test = np.mean((test_target - test_pred) ** 2)
print("Test Mean Squared Error:", mse_test)   

# graph testing results
plt.plot(sim.trange()[t_test], test_pred, label="prediction")
plt.plot(sim.trange()[t_test], test_target, label="target")
plt.title("Testing Output")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
plt.show() 