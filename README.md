# 🛒 Rossmann Store Sales Forecasting

This project tackles the Rossmann Store Sales Kaggle competition, which involves predicting daily sales for over 1,000 stores across several years of historical data. The challenge focuses on applying time series analysis, feature engineering, and machine learning to forecast sales accurately.

## Dataset

- Source: [Kaggle Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- Data files:
  - `train.csv`: Historical daily sales data per store
  - `test.csv`: Data to predict sales for
  - `store.csv`: Metadata about each store (type, competition, promo info)

## Feature Engineering

- **Lag Features**: `Sales_lag_1`, `Sales_lag_7`
- **Rolling Statistics**: Mean and Std for 7 and 14 days
- **Date Features**: Year, Month, Week, Day, IsWeekend
- **Categorical Encoding**:
- `StateHoliday`, `StoreType`, `Assortment`, and `PromoInterval` one-hot encoded
- **Missing Values**: Filled using median from training set

## Models Used

### 🔹 Baseline Models
These simple models help set a benchmark for more complex ones:
- **Naive Forecast** – Predicts today’s sales to be the same as yesterday’s.
- **Moving Average** – Predicts using the average of the past few days.

### 🔹 Exponential Smoothing Models (ETS)
Traditional time series models for trend/seasonality:
- **Simple Exponential Smoothing (SES)**
- **Holt’s Linear Trend Method (DES)**
- **Holt-Winters Additive Method (TES)**
- **Holt-Winters Multiplicative Method (TES-Mul)**

### 🔹 Statistical Forecasting Models
Advanced time series techniques:
- **ARIMA** – AutoRegressive Integrated Moving Average
- **SARIMAX** – Seasonal ARIMA with exogenous regressors

### 🔹 Facebook Prophet
Flexible time series model with holidays and seasonality:
- **Prophet**
- **Prophet with Regressors** – Includes external variables like promotions, etc.

### 🔹 Machine Learning Models
Tree-based regressors trained on engineered features:
- **Gradient Boosting Regressor (GBR)**
- **XGBoost Regressor (XGB)**
- **CatBoost Regressor (CBR)**

  
## Tech Stack

- **Python** 
- **Pandas**, **NumPy** – Data wrangling and manipulation
- **Matplotlib**, **Seaborn**, **Plotly** – Visualizations
- **Scikit-learn** – ML preprocessing and evaluation
- **Statsmodels**, **Prophet**, **ARIMA** – (optional for time series models)
- **Jupyter Notebook** – Development and exploration

## 📅 Time Series Split

Instead of random train-test split, we used **chronological split**:
- 90% earliest data for training
- 10% most recent data for validation
This simulates a real-world forecasting scenario.

```python
# Chronological split
split_date = train["Date"].quantile(0.9)
train_data = train[train["Date"] < split_date]
valid_data = train[train["Date"] >= split_date]
