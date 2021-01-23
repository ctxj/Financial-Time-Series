#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

#%%
df = pd.read_csv('C:\\Users\\Caspar\\Desktop\\python_project\\time series\\fb_rsi.csv')
#%%
close = np.array(df['close'])
date = np.array(df['date'])
print(len(close))
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
    tf.keras.layers.Lambda(lambda x: x*500)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
# %%
history = model.fit(train_set, epochs=500)
#%%
mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss))

plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.show()
#%%
plt.plot(epochs[300:], mae[300:], 'r')
plt.plot(epochs[300:], loss[300:], 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.show()
#%%
plt.plot(epochs[100:200], mae[100:200], 'r')
plt.plot(epochs[100:200], loss[100:200], 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.show()
#%%
model.save('lstm.h5')
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
plt.title('Actual vs Forcast')
plt.legend(['Actual', 'Forecast'])
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
pred = ndf['forecast'].pct_change()
rsi = ndf['rsi']
close = ndf['close']
#%%
pred
# %%
cond1 = (pred < 0) & (pred.shift(-1) > 0)
cond2 = (cond1 == True) & (pred.shift(-1) > 0)
cond3 = (cond2 == True) & (cond2.shift(1) == False)
cond4 = (cond2 == False) & (cond2.shift(1) == True)

cond5 = (pred > 0) & (pred.shift(-1) < 0)
cond6 = (cond5 == True) & (pred.shift(-1) < 0)
cond7 = (cond6 == True) & (cond6.shift(1) == False)
cond8 = (cond6 == False) & (cond6.shift(1) == True)


ml_enterlong = close[cond1]
ml_exitlong = close[cond2]

ml_entershort = close[cond7]
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
ndf.to_csv('ml_model.csv', index=False)
# %%
