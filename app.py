import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.title("📈 Stock Price Prediction ")

# User input
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, RELIANCE.NS)").upper()
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.Timestamp.today().date())

if stock and start_date and end_date:

    df = yf.download(stock, start=start_date, end=end_date)

    # 🔥 Fix duplicate / MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    if df.empty:
        st.warning("No data found for this stock symbol or date range!")
        st.stop()

    st.subheader("Stock Data")
    st.dataframe(df.tail())

    # Load model
    model = load_model("model/stock_model.keras")

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = df[['Close']].copy()
    scaled_data = scaler.fit_transform(close_prices)

    # Ensure enough data
    if len(scaled_data) < 60:
        st.error("Not enough data to predict. Need at least 60 rows.")
        st.stop()

    last_60_days = scaled_data[-60:]
    current_input = last_60_days.copy()

    predictions = []

    # Predict 7 days
    for _ in range(7):
        input_reshaped = current_input.reshape(1, 60, 1)
        pred = model.predict(input_reshaped, verbose=0)[0][0]
        predictions.append(pred)

        current_input = np.vstack((current_input[1:], [[pred]]))

    # Inverse transform
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Safe float extraction
    last_price = float(df['Close'].to_numpy().flatten()[-1])

    st.subheader("Results")
    st.write(f"Last Actual Price: {last_price:.2f}")

    st.subheader("Next 7 Days Predicted Prices")
    for i, price in enumerate(predictions, 1):
        st.write(f"Day {i}: {float(price[0]):.2f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Actual Prices')

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
    plt.plot(future_dates, predictions, marker='o', label='Predicted Prices')

    plt.title(f"{stock} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)