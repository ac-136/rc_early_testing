# Try to use reservior computing on the airplane dataset from sktime
## Details:
* Analyzing Box & Jenkins airline data. Monthly totals of international airline passengers, 1949 to 1960
* my_reservoir.py is based on: https://github.com/stevenabreu7/handson_reservoir/blob/main/reservoir.py
* For main.py
    * Data preparation
        * lags = 12
    * Reservoir parameters
        * n_neurons = 20
        * rhow (spectral radius) = 1.25
        * leak_range = (0.1, 0.3)
## How to run:
```sh
python main.py
```