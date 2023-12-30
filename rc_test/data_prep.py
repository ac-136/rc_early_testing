
def make_lags(df, num_lags):
    '''
        Function used to add lags to pandas dataframe

        df: dataframe to add lags to; column you want to lag must be at col index 0
        num_lags: number of lag features to add to dataframe
    '''
    for i in range(1, num_lags+1):
        lag_name = f'lag_{i}'
        df[lag_name] = df.iloc[:, 0].shift(i)
