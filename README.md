# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 1.11.2025
### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# === 1. Load Dataset ===
data = pd.read_csv("/content/Clean_Dataset.csv")

# === 2. Prepare Dataset ===
print("Columns in dataset:", data.columns.tolist())

# Ensure required columns exist
if 'days_left' not in data.columns:
    raise ValueError("Column 'days_left' not found in dataset!")
if 'price' not in data.columns:
    raise ValueError("Column 'price' not found in dataset!")

# Group by days_left and calculate average price per day
daily_avg = data.groupby('days_left')['price'].mean().sort_index()

# Convert to a pandas time series (days_left treated as pseudo-date)
ts_data = pd.Series(daily_avg.values, index=pd.RangeIndex(len(daily_avg)))
ts_data.index.name = 'DayIndex'

# === 3. Display Info ===
print("Shape of dataset:", ts_data.shape)
print("\nFirst 10 rows:")
print(ts_data.head(10))

# === 4. Plot Original Data ===
plt.figure(figsize=(12, 6))
plt.plot(ts_data, label='Average Price by Days Left', color='blue')
plt.title('Flight Price Trend vs Days Left')
plt.xlabel('Days Left Index')
plt.ylabel('Average Price')
plt.legend()
plt.grid()
plt.show()

# === 5. Moving Averages ===
rolling_mean_5 = ts_data.rolling(window=5).mean()
rolling_mean_10 = ts_data.rolling(window=10).mean()

# Plot moving averages
plt.figure(figsize=(12, 6))
plt.plot(ts_data, label='Original', color='blue')
plt.plot(rolling_mean_5, label='MA (5)', color='orange')
plt.plot(rolling_mean_10, label='MA (10)', color='green')
plt.title('Moving Averages (Flight Prices)')
plt.xlabel('Days Left Index')
plt.ylabel('Average Price')
plt.legend()
plt.grid()
plt.show()

# === 6. Scale Data ===
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(ts_data.values.reshape(-1, 1)).flatten(),
    index=ts_data.index
)
scaled_data = scaled_data + 1e-3  # stability for multiplicative models

# === 7. Train-Test Split ===
split_index = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split_index]
test_data = scaled_data[split_index:]

# === 8. Holt-Winters Model ===
model = ExponentialSmoothing(
    train_data, trend='add', seasonal='mul', seasonal_periods=5
).fit()

predictions = model.forecast(steps=len(test_data))

# Plot Results
ax = train_data.plot(label='Train', figsize=(12, 6))
test_data.plot(ax=ax, label='Test')
predictions.plot(ax=ax, label='Predictions', color='red')
plt.title('Holt-Winters Forecast (Flight Prices by Days Left)')
plt.xlabel('Days Left Index')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid()
plt.show()

# === 9. RMSE Evaluation ===
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# === 10. Forecast Future Prices ===
model_full = ExponentialSmoothing(
    scaled_data, trend='add', seasonal='mul', seasonal_periods=5
).fit()

future_steps = 15  # Forecast next 15 pseudo-days
future_forecast = model_full.forecast(steps=future_steps)

# Plot future forecast
ax = scaled_data.plot(label='Historical', figsize=(12, 6))
future_forecast.plot(ax=ax, label='Forecast', color='red')
plt.title('Forecast of Future Flight Prices (Next 15 Days)')
plt.xlabel('Days Left Index')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
<img width="1335" height="718" alt="image" src="https://github.com/user-attachments/assets/be2f2d0d-3406-494d-9b23-63e2781d7352" />

<img width="1276" height="545" alt="image" src="https://github.com/user-attachments/assets/09bd41b0-6c5d-481c-8374-f3fe62b1a986" />

<img width="1218" height="528" alt="image" src="https://github.com/user-attachments/assets/e955703f-87c6-4127-b358-b5b4b9928d19" />

<img width="1141" height="514" alt="image" src="https://github.com/user-attachments/assets/71ce3041-62c6-40ca-8397-0245d9cc17e0" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
