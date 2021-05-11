# Financial Time Series Analysis
Used deep learning models to perform time series analysis on financial data  
Developed trading strategies using models forecasts  
Compared strategies with non ML approach

## Prerequisites
Python 3.8.6  
[Python libraries](https://github.com/ctxj/Financial-Time-Series/blob/main/requirements.txt) pip install -r requirements.txt  aa
[IEX api key](https://iexcloud.io/core-data/)

## Data
Facebook's intraday data from IEX api  
[intraday_data.py](https://github.com/ctxj/Financial-Time-Series/blob/main/intraday_data.py)

## Models
[LSTM](https://github.com/ctxj/Financial-Time-Series/blob/main/lstm_fb.ipynb) Best performing model  
[LSTM Strategy](https://github.com/ctxj/Financial-Time-Series/blob/main/lstm_vs_rsi.ipynb) Backtest LSTM strategy  
[DNN](https://github.com/ctxj/Financial-Time-Series/blob/main/fb_models.ipynb) Model overfit  
[RSI strategy](https://github.com/ctxj/Financial-Time-Series/blob/main/rsi_fb.ipynb) Non ML model as baseline

## Results
![LSTM VS RSI](https://raw.githubusercontent.com/ctxj/Financial-Time-Series/main/img/results.png)

## License
![Apache License 2.0](https://img.shields.io/badge/License-Apache--License--2.0-yellow.svg)
