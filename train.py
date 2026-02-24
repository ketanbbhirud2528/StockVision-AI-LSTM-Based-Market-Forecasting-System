import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import os

# Create model folder if not exists
if not os.path.exists("model"):
    os.makedirs("model")

# Download Data (Apple example)
data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")

close_data = data["Close"].values.reshape(-1,1)

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

# Create dataset
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# Train Model
model.fit(X, y, epochs=5, batch_size=32)

# Save Model
model.save("model/stock_model.keras")

print("Model Trained and Saved Successfully ✅")