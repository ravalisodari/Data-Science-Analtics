# 📊 Data Science And Analytic
# Time Series Forecasting for Stock Prices

This project demonstrates and compares several popular time series forecasting models to predict the future stock price of Apple Inc. (AAPL). The goal is to apply ARIMA, SARIMA, Prophet, and LSTM models to historical stock data and evaluate their performance based on the Root Mean Squared Error (RMSE).

The entire workflow, from data acquisition to model evaluation, is contained within the `TimeSeriesForecasting.ipynb` notebook. The project culminates in a simple Streamlit dashboard to visualize and compare the results.

## Forecasting Models & Results

### 1. Historical Stock Price Data (AAPL)
First, we download the historical stock price data for AAPL from 2015 to the end of 2024 using the `yfinance` library. We focus on the 'Close' price for our analysis.



### 2. ARIMA Model Forecast
The Autoregressive Integrated Moving Average (ARIMA) model is a classic statistical model for time series analysis. We use an `ARIMA(5,1,0)` model, which considers the 5 previous time steps for autoregression, uses a first-order differencing to make the series stationary, and has no moving average window.



### 3. SARIMA Model Forecast
The Seasonal ARIMA (SARIMA) model extends ARIMA by adding a seasonal component. This is particularly useful for time series with cyclical patterns. We use a `SARIMA(1,1,1)x(1,1,0,12)` model, which adds a yearly seasonality component (12 months). The plot shows the forecast along with the confidence interval.



### 4. Prophet Model Forecast
Prophet, developed by Facebook, is a powerful and easy-to-use forecasting tool that is robust to missing data and shifts in the trend. It automatically handles weekly and yearly seasonality.



### 5. LSTM Model Forecast
Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) well-suited for time series forecasting due to its ability to remember patterns over long periods. We built a simple sequential model with two LSTM layers and a Dense output layer.



---

## Model Performance Comparison

The models are evaluated on their ability to predict future data points. The Root Mean Squared Error (RMSE) is used as the primary metric to measure accuracy.

### RMSE Scores

| Model   | RMSE      |
|---------|-----------|
| ARIMA   | 116.34    |
| SARIMA  | 116.35    |
| Prophet | 118.84    |
| LSTM    | 5.53      |

### Visual Comparison

To better visualize the performance difference, the following pie chart represents the models' **proportional accuracy**. Since a lower RMSE is better, accuracy is calculated as `1/RMSE`. A larger slice in the chart indicates a better-performing model.



**Conclusion:** The **LSTM model**, with an RMSE of just **5.53**, significantly outperforms the other models. This is visually represented by its dominant share (over 80%) of the proportional accuracy pie chart. The traditional statistical models (ARIMA, SARIMA, Prophet) perform similarly to each other but are far less accurate than the deep learning approach for this specific dataset.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Pip package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ravalisodari/Data-Science-Analtics.git
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is provided for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

---
### `requirements.txt`
```
numpy
pandas
scikit-learn
matplotlib
yfinance
statsmodels
prophet
tensorflow
streamlit
```
---
## Usage

1.  **Run the Jupyter Notebook:**
    Open and run the cells in `TimeSeriesForecasting.ipynb` to see the data processing, model training, and individual plot generation.
    ```bash
    jupyter notebook TimeSeriesForecasting.ipynb
    ```

2.  **Generate the Performance Chart:**
    To generate the pie chart comparing model performance, run the following script. This will save the chart as `rmse_pie_chart.png` in your project directory.

    **`generate_chart.py`:**
    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    # Data from the notebook
    models = ['ARIMA', 'SARIMA', 'Prophet', 'LSTM']
    rmse_values = [116.34, 116.35, 118.84, 5.53]

    # Invert RMSE for accuracy score (lower RMSE is better, so higher 1/RMSE is better)
    accuracy_scores = 1 / np.array(rmse_values)

    # Explode the best performing model (LSTM)
    explode = (0, 0, 0, 0.1)

    # Create labels with model names and their RMSEs
    labels = [f'{model}\n(RMSE: {rmse:.2f})' for model, rmse in zip(models, rmse_values)]

    # Create the pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(accuracy_scores, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12})
    plt.title('Model Performance Comparison (Proportional Accuracy based on 1/RMSE)', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save the figure
    plt.savefig('rmse_pie_chart.png', bbox_inches='tight')
    print("Pie chart saved as rmse_pie_chart.png")
    ```
    Run from your terminal:
    ```bash
    python generate_chart.py
    ```
---
## Code Improvements & Fixes

The original notebook has a few areas that were corrected or could be improved for clarity and correctness:

1.  **Module Installation**: The first cell fails due to a `ModuleNotFoundError`. The notebook should begin with `!pip install yfinance` to make it self-contained, or users should be instructed to use the `requirements.txt` file.

2.  **Code Redundancy in LSTM Cells**: The notebook contains a cell with dummy data for the LSTM and another with the actual implementation, which is a bit confusing. The final LSTM cell was also a mix of training, testing, and plotting logic. This has been cleaned up for clarity.

3.  **SARIMA Plotting Bug**: In the original notebook, the `forecast` variable from the *ARIMA* model is used for plotting instead of the `forecast_sarima` variable. The plot should use the correct forecast object:
    ```python
    # Original buggy line:
    # forecast.plot(label='Forecast', color='green')

    # Corrected line:
    forecast_sarima.plot(label='Forecast', color='green')
    ```

4.  **LSTM Logic and Plotting**: The LSTM implementation in the notebook had several issues with data splitting, sequence creation for testing, and plotting indices. The final LSTM cell was refactored to correctly:
    - Split data into training and testing sets.
    - Create sequences for the test set (`X_test_seq`, `y_test_actual`).
    - Make predictions on the test set.
    - Correctly calculate the RMSE using the inverse-transformed actuals and predictions.
    - Plot the predictions against the correct segment of the original data.
