# Financal Time Series Analysis
Use deep learning models to perform time series analysis on financial data  
Develop trading strategies using models forcasts  
Compare stratergies with non ML approach

## Prerequisites
---
Python 3.8.6  
[Python libraries](https://github.com/ctxj/Financial-Time-Series/blob/main/requirements.txt) pip install -r requirements.txt  
[IEX api key](https://iexcloud.io/core-data/)

## Data
---
Facebook's intraday data from IEX api  
[intraday_data.py](https://github.com/ctxj/Financial-Time-Series/blob/main/intraday_data.py)

## Models
---
[LSTM](https://github.com/ctxj/Financial-Time-Series/blob/main/lstm_fb.ipynb) Best performing model  
[DNN](https://github.com/ctxj/Financial-Time-Series/blob/main/fb_models.ipynb) Model overfit  
[RSI strategy](https://github.com/ctxj/Financial-Time-Series/blob/main/rsi_fb.ipynb) non ML model as baseline

## Results
---
[LSTM VS RSI](https://raw.githubusercontent.com/ctxj/Financial-Time-Series/main/img/results.png)

## License
---
[Apache License 2.0](https://github.com/ctxj/Financial-Time-Series/blob/main/LICENSE)