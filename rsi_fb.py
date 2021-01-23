#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

# %%
df = pd.read_csv('C:\\Users\\Caspar\\Desktop\\python_project\\time series\\FB_raw.csv')
df['date'] = pd.to_datetime(df['date'])
df['close'].fillna(method='ffill', inplace=True)

locator = mdates.MonthLocator()

plt.plot(df['date'], df['close'])
plt.xticks(rotation=45)
x = plt.gca()
x.xaxis.set_major_locator(locator)
x.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
plt.show()
#%%
period = 5
close = df['close']
delta = close.diff()

up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0

rolling_gain = up.ewm(span=period).mean()
rolling_loss = down.abs().ewm(span=period).mean()

rs = rolling_gain/rolling_loss
rsi = 100 - (100 / (1+rs))

df['rsi'] = rsi
 # %%
cond1 = (rsi <= 30) & (rsi.shift(1) > 30)
cond2 = (cond1 == True) & (rsi <= 70)
cond3 = (cond2 == True) & (cond2.shift(1) == False)
cond4 = (cond2 == False) & (cond2.shift(1) == True)

cond5 = (rsi >= 70) & (rsi.shift(1) < 70)
cond6 = (cond5 == True) & (rsi >= 30)
cond7 = (cond6 == True) & (cond6.shift(1) == False)
cond8 = (cond6 == False) & (cond6.shift(1) == True)

enterlong = close[cond3]
exitshort = close[cond4]

entershort = close[cond7]
exitlong = close[cond8]
# %%
df['long'] = enterlong
df['exit_long'] = exitlong

df['short'] = entershort
df['exit_short'] = exitshort
# %%
df['exit_long'].fillna(method='bfill', inplace=True)
long_pnl = df['exit_long'].shift(-1) - df['long']
df['long_pnl'] = long_pnl

df['exit_short'].fillna(method='bfill', inplace=True)
short_pnl = df['short'] - df['exit_short'].shift(-1)
df['short_pnl'] = short_pnl

df['total_pnl'] = df[['short_pnl', 'long_pnl']].sum(axis=1)
#%%
print(f"Total number of long trades: {df['long_pnl'].count()}")
print(f"Total number of short trades: {df['short_pnl'].count()}")
# %%
performance_hold = df['close'].iloc[-1] - df['close'].iloc[1]
performance_long = df['long_pnl'].cumsum().fillna(method='ffill')
performance_short = df['short_pnl'].cumsum().fillna(method='ffill')
total_performance = df['total_pnl'].cumsum().fillna(method='ffill')

print(f"Profit from hold strategy: {performance_hold}")
print(f"Profit from short strategy: {(df['short_pnl'].sum())}")
print(f"Profit from long strategy: {df['long_pnl'].sum()}")
print(f"Profit from long & short strategies: {(df['total_pnl'].sum())}")
print(f"Max long draw down: {performance_long.min()}")
print(f"Max short draw down: {performance_short.min()}")
print(f"Max total draw down: {total_performance.min()}")

locator = mdates.MonthLocator()
fmt = mdates.DateFormatter('%b %y')

plt.plot(df['date'], df['close']-167.925, alpha=0.3)
plt.plot(df['date'], performance_long, alpha=0.5)
plt.plot(df['date'], performance_short, alpha=0.5, color='red')
plt.plot(df['date'], total_performance, color='green')
x = plt.gca()
x.xaxis.set_major_locator(locator)
x.xaxis.set_major_formatter(fmt)
plt.xticks(rotation=45)
plt.show()

# %%
df['long_labels'] = np.where(df['long_pnl'] <= 0, 0,
                        np.where(df['long_pnl'] > 0, 1, np.nan))

df['short_labels'] = np.where(df['short_pnl'] <= 0, 0, 
                        np.where(df['short_pnl'] > 0, 1, np.nan))
#%%
print(df['long_labels'].value_counts())
print(df['short_labels'].value_counts())
# %%
df.to_csv('fb_rsi4.csv', index=False)
# %%
close = np.array(df['close'])
date = np.array(df['date'])
# %%
split = 100000
date_train = date[:split]
x_train = close[:split]
date_val = date[split:]
x_val = close[split:]

window_size = 5
batch_size = 250
shuffle_buffer_size = 10000
# %%
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
# %%
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                            strides=1, padding='causal',
                            activation='relu', input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*400)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
# %%
history = model.fit(train_set, epochs=500)
# %%
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(250).prefetch(1)
    forecast = model.predict(ds)
    return forecast
# %%
forecast = model_forecast(model, close[..., np.newaxis], window_size)
forecast = forecast[split - window_size:-1, -1, 0]
# %%
locator = mdates.MonthLocator()
fmt = mdates.DateFormatter('%b %y')

plt.plot(date_val, x_val, alpha=0.5)
plt.plot(date_val, forecast)

x = plt.gca()
x.xaxis.set_major_locator(locator)
x.xaxis.set_major_formatter(fmt)
plt.xticks(rotation=45)
plt.show()
# %%
tf.keras.metrics.mean_absolute_error(x_val, forecast).numpy()
# %%
forecast
# %%
ndf = df.iloc[split:]
ndf.count()
# %%
ndf['forecast'] = np.float64(forecast)
pred = ndf['forecast']
#%%
close = ndf['close']
# %%
cond1 = (pred > close) & (pred > pred.shift(1))
cond2 = (pred < close)
cond3 = (pred < close) & (pred < pred.shift(1))
cond4 = (pred > close)


ml_enterlong = close[cond1]
ml_exitlong = close[cond2]

ml_entershort = close[cond3]
ml_exitshort = close[cond4]
#%%
ndf['ml_long'] = ml_enterlong
ndf['ml_exit_long'] = ml_exitlong

ndf['ml_short'] = ml_entershort
ndf['ml_exit_short'] = ml_exitshort
# %%
ndf['ml_exit_long'].fillna(method='bfill', inplace=True)
ml_long_pnl = ndf['ml_exit_long'].shift(-1) - ndf['ml_long']
ndf['ml_long_pnl'] = ml_long_pnl

ndf['ml_exit_short'].fillna(method='bfill', inplace=True)
ml_short_pnl = ndf['ml_short'] - ndf['ml_exit_short'].shift(-1)
ndf['ml_short_pnl'] = ml_short_pnl

ndf['ml_total_pnl'] = ndf[['ml_short_pnl', 'ml_long_pnl']].sum(axis=1)
# %%
print(ndf['ml_long_pnl'].count())
print(ndf['ml_short_pnl'].count())
print(ndf['long_pnl'].count())
print(ndf['short_pnl'].count())
# %%
performance_long = ndf['long_pnl'].cumsum().fillna(method='ffill')
performance_short = ndf['short_pnl'].cumsum().fillna(method='ffill')
total_performance = ndf['total_pnl'].cumsum().fillna(method='ffill')

ml_performance_long = ndf['ml_long_pnl'].cumsum().fillna(method='ffill')
ml_performance_short = ndf['ml_short_pnl'].cumsum().fillna(method='ffill')
ml_total_performance = ndf['ml_total_pnl'].cumsum().fillna(method='ffill')

print(f"Profit from short strategy: {(ndf['short_pnl'].sum())}")
print(f"Profit from long strategy: {ndf['long_pnl'].sum()}")
print(f"Profit from both strategies: {(ndf['total_pnl'].sum())}")
print(f"Max long draw down: {performance_long.min()}")
print(f"Max short draw down: {performance_short.min()}")
print(f"Max total draw down: {total_performance.min()}")

print(f"Profit from short strategy: {(ndf['ml_short_pnl'].sum())}")
print(f"Profit from long strategy: {ndf['ml_long_pnl'].sum()}")
print(f"Profit from both strategies: {(ndf['ml_total_pnl'].sum())}")
print(f"Max long draw down: {ml_performance_long.min()}")
print(f"Max short draw down: {ml_performance_short.min()}")
print(f"Max total draw down: {ml_total_performance.min()}")

locator = mdates.MonthLocator()
fmt = mdates.DateFormatter('%b %y')

plt.plot(ndf['date'], ndf['close'], alpha=0.2)
plt.plot(ndf['date'], performance_long, alpha=0.5, color='cyan')
plt.plot(ndf['date'], performance_short, alpha=0.5, color='orange')
plt.plot(ndf['date'], total_performance, alpha=0.8, color='lightgreen')
plt.plot(ndf['date'], ml_performance_long, alpha=0.7, color='blue')
plt.plot(ndf['date'], ml_performance_short, alpha=0.7, color='red')
plt.plot(ndf['date'], ml_total_performance, color='green')
x = plt.gca()
x.xaxis.set_major_locator(locator)
x.xaxis.set_major_formatter(fmt)
plt.xticks(rotation=45)
plt.show()
# %%
