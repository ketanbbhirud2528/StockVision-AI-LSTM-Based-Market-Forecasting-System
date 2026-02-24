import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

model = load_model("model/stock_model.h5")

stock = "AAPL"
df = yf.download(stock, start="2015-01-01", end="2025-01-01")

data = df[['Close']]
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

last_60_days = scaled_data[-60:]
X_test = []
X_test.append(last_60_days)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

prediction = model.predict(X_test)
prediction = scaler.inverse_transform(prediction)

print("Next Day Predicted Price:", prediction[0][0])