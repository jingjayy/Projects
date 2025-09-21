import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import seaborn as sns
import os

# For ARIMA model
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm

# For Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# For LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam 

# ======================================================
# PART 1: DATASET ACQUISITION FOR CASES_MALAYSIA.CSV
# ======================================================
df_country = pd.read_csv("D:\AI class\directory\capstone2_folder\cases_malaysia.csv")
print(df_country.head(10))

# =======================================
# PART 2: EXPLORATORY DATASET ANALYSIS
# =======================================
# 2.1 DATA PROCESSING (Preliminary for EDA)
# This section remains focused on getting the data ready for EDA.
print("\n--- 2.1 Data Processing (Preliminary for EDA) ---")
df_country['date'] = pd.to_datetime(df_country['date'])
df_country.set_index('date', inplace=True)

# Apply data filtering (after January 2022)
df_national_filtered = df_country[df_country.index >= '2022-01-01'].copy()
print(f"Data filtered from {df_national_filtered.index.min().strftime('%Y-%m-%d')} to {df_national_filtered.index.max().strftime('%Y-%m-%d')}.")

# Select the target variable for EDA
ts_data = df_national_filtered['cases_new'].copy()

# Check for missing values in cases_new after filtering
# No missing values should be expected.
missing_values_count = ts_data.isnull().sum()
if missing_values_count > 0:
    print(f"Found {missing_values_count} missing values in 'cases_new' after filtering. Forward filling.")
    ts_data = ts_data.ffill()
else:
    print("No missing values found in 'cases_new' after filtering")

# Apply 7-day moving average for smoothing
# Create a smoothed version, but for initial EDA plots, crucial for mitigating the impact of the weekly seasonality for 
# focusing on ARIMA only models
ts_data_smoothed = ts_data.rolling(window=7, center=False).mean().ffill().bfill()
print("Smoothed 'ts_data_smoothed' series created using 7-day moving average, specifically for non-seasonal ARIMA.")

# 2.2 SUMMARY STATISTICS
print("\n========== 2.2 SUMMARY STATISTICS ==========")

# 2.2.1 Numerical Summary (.describe())
print("\n--- Numerical Summary for 'cases_new' (January 2022 onwards) ---")
print(ts_data.describe())

# Additional checks for 0 cases
zero_cases_count = (ts_data == 0).sum()
print(f"\nNumber of days with 0 new cases: {zero_cases_count}")
print(f"Percentage of days with 0 new cases: {zero_cases_count / len(ts_data) * 100:.2f}%")

# 2.2.2 Histogram of Daily New Cases
plt.figure(figsize=(12, 5))
sns.histplot(ts_data, bins=50, kde=True)
plt.title('Distribution of Daily New COVID-19 Cases (Jan 2022 onwards)')
plt.xlabel('New Cases')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('national_eda_histogram.png')
plt.show()

# 2.2.3 Box Plot of Daily New Cases
plt.figure(figsize=(12, 5))
sns.boxplot(x=ts_data)
plt.title('Box Plot of Daily New COVID-19 Cases (Jan 2022 onwards)')
plt.xlabel('New Cases')
plt.grid(axis='x', alpha=0.75)
plt.savefig('national_eda_boxplot.png')
plt.show()

# 2.3 MISSING VALUES & ANOMALIES
print("\n========== 2.3 MISSING VALUES & ANOMALIES ==========")

# 2.3.1 Missing Values Check (Re-confirmation)
print("\n--- Missing Values Check for 'cases_new' ---")
if missing_values_count > 0:
    print(f"Warning: Found {missing_values_count} missing values.")
else:
    print("No missing values found in 'cases_new', confirming data quality.")

# 2.3.2 High Outliers Detection & Visualization
# Purpose: Quantifies and visualizes extreme spikes as anomalies.
Q1 = ts_data.quantile(0.25)
Q3 = ts_data.quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_upper = Q3 + 1.5 * IQR
# the current value is more than the threshold then it will be put onto high outlier
outliers_high = ts_data[ts_data > outlier_threshold_upper]

print(f"\nUpper outlier threshold (Q3 + 1.5*IQR): {outlier_threshold_upper:.2f}")
print(f"Number of data points above this threshold: {len(outliers_high)}")
if not outliers_high.empty:
    print("Dates with high outlier cases (first 5):")
    print(outliers_high.head())
else:
    print("No high outliers detected above the 1.5*IQR threshold.")

plt.figure(figsize=(14, 7))
plt.plot(ts_data.index, ts_data, label='Daily New Cases')
if not outliers_high.empty:
    plt.scatter(outliers_high.index, outliers_high, color='red', marker='o', label='High Outliers Detected (1.5*IQR)')
plt.title('Daily New COVID-19 Cases with Highlighted High Outliers (Jan 2022 onwards)')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True)
plt.savefig('national_eda_high_outliers.png')
plt.show()

# 2.4 TREND ANALYSIS
print("\n========== 2.4 TREND ANALYSIS ==========")

# --- Overall Time Series Line Plot (Original 'df_national_filtered' data) ---
# Purpose: Fundamental plot to observe the overall trend and major shifts
plt.figure(figsize=(15, 7))
sns.lineplot(x=df_national_filtered.index, y='cases_new', data=df_national_filtered, color='skyblue')
plt.title('Daily New COVID-19 Cases Over Time (Jan 2022 onwards)')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('national_eda_overall_trend.png')
plt.show()

# --- Rolling Statistics (Mean & Standard Deviation) Plot ---
# Purpose: Visualizes underlying trends and changes in variance.
rolling_mean = ts_data.rolling(window=7).mean()
rolling_std = ts_data.rolling(window=7).std()

plt.figure(figsize=(16, 8))
plt.plot(ts_data, label='Original Daily New Cases', alpha=0.7)
plt.plot(ts_data_smoothed, color='red', label='Causal 7-Day Moving Average (for ARIMA)')
plt.plot(rolling_std, color='black', label='7-Day Rolling Std Dev')
plt.title('Daily New COVID-19 Cases with 7-Day Rolling Statistics (Jan 2022 onwards)')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True)
plt.savefig('national_eda_rolling_stats.png')
plt.show()

# --- Seasonal Decomposition Plot ---
# Purpose: Formally separates Trend, Seasonal, and Residual components, crucial for
# understanding seasonality and overall temporal characteristics.
seasonal_period = 7 # Weekly seasonality for daily data
if len(ts_data) >= seasonal_period * 2:
    decomposition = seasonal_decompose(ts_data, model='additive', period=seasonal_period, extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle(f'Time Series Decomposition (Seasonal Period = {seasonal_period}) - Additive Model\n(Jan 2022 onwards)', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('national_eda_seasonal_decomposition.png')
    plt.show()
else:
    print(f"\nNot enough data for seasonal decomposition with period {seasonal_period}.")


# --- Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) Plots (Original Series) ---
# Purpose: Fundamental for understanding the internal correlation structure (trend and seasonality).
print("\n--- 5.2 ACF/PACF Plots of Smoothed National Data (for understanding autocorrelation structure) ---")
lags_to_plot = 50
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
plot_acf(ts_data_smoothed, lags=lags_to_plot, ax=axes[0], title='Autocorrelation Function (ACF) - Causal Smoothed Series')
plot_pacf(ts_data_smoothed, lags=lags_to_plot, ax=axes[1], title='Partial Autocorrelation Function (PACF) - Causal Smoothed Series')
plt.tight_layout()
plt.savefig('arima_eda_acf_pacf_smoothed_for_model.png') # New filename to distinguish
plt.show()

# --- Lag Plot (Lag 1 & Lag 7) ---
# Purpose: Visualizes direct relationships between current and past values to confirm autocorrelation and seasonality.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
pd.plotting.lag_plot(ts_data_smoothed, lag=1, ax=axes[0])
axes[0].set_title('Lag Plot (Lag 1) - Causal Smoothed Series')
axes[0].grid(True)
pd.plotting.lag_plot(ts_data_smoothed, lag=7, ax=axes[1])
axes[1].set_title('Lag Plot (Lag 7 - Weekly) - Causal Smoothed Series')
axes[1].grid(True)
plt.tight_layout()
plt.savefig('national_eda_lag_plot_smoothed.png')
plt.show()
print("Plots will help in visualizing the temporal dependencies in the smoothed series,")
print("informing the 'p' and 'q' parameters that 'auto_arima' will identify for the non-seasonal ARIMA model.")

# --- Average New Cases by Day of Week Bar Chart ---
# Purpose: Explicitly visualizes the weekly seasonality pattern.
ts_data_dayofweek_ori = ts_data.groupby(ts_data.index.dayofweek).mean() # <<< CHANGE 'ts_data_smoothed' to 'ts_data'
ts_data_dayofweek_ori.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(10, 6))
ts_data_dayofweek_ori.plot(kind='bar')
plt.title('Average New Cases by Day of Week (Original Series, Jan 2022 onwards)')
plt.xlabel('Day of Week')
plt.ylabel('Average New Cases') # Keep original label as it's for raw data
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('national_eda_avg_dayofweek_original.png')
plt.show()

# --- Average Seasonal Component by Day of Week Bar Chart (from Decomposition) ---
# This plot directly visualizes the original weekly pattern found by seasonal_decompose,
# it clearly showing what the 7-day MA for ARIMA helps to mitigate. 
# This plot is used to show the consistent periodic seasonality.
if len(ts_data) >= seasonal_period * 2: # Check if decomposition was successful
    seasonal_pattern_by_day = pd.Series(decomposition.seasonal, index=ts_data.index).groupby(ts_data.index.dayofweek).mean()
    seasonal_pattern_by_day.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    plt.figure(figsize=(10, 6))
    seasonal_pattern_by_day.plot(kind='bar', color='purple')
    plt.title('Average Seasonal Component by Day of Week (from Original Data Decomposition)')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Seasonal Effect')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('national_eda_avg_seasonal_component_dayofweek.png')
    plt.show()

# --- Year-over-Year Comparison Plot ---
# Purpose: Compares annual patterns within the post-pandemic phase.
plt.figure(figsize=(16, 8))
for year in ts_data_smoothed.index.year.unique():
    yearly_data = ts_data_smoothed.loc[ts_data_smoothed.index.year == year]
    plt.plot(yearly_data.index.dayofyear, yearly_data, label=str(year), alpha=0.7)
plt.title('New Cases: Year-over-Year Comparison (Jan 2022 onwards, Causal Smoothed by Day of Year)')
plt.xlabel('Day of Year')
plt.ylabel('New Cases (7-day MA)')
plt.legend()
plt.grid(True)
plt.savefig('national_eda_yoy_comparison_smoothed.png')
plt.show()

# ==================================================
# PART 3: DATASET ACQUISITION FOR CASES_STATE.CSV
# ==================================================
df_state = pd.read_csv("D:\AI class\directory\capstone2_folder\cases_state.csv")
print(df_state.head(10))

# =======================================
# PART 4: EXPLORATORY DATASET ANALYSIS
# =======================================
# 4.1 DATA PROCESSING (Preliminary for EDA)
# This section remains focused on getting the data ready for EDA.
print("\n--- 4.1 Data Processing (Preliminary for EDA) ---")
df_state['date'] = pd.to_datetime(df_state['date'])
df_state.set_index('date', inplace=True)

# Apply data filtering (after January 2022)
df_state_filtered = df_state[df_state.index >= '2022-01-01'].copy()
print(f"Data filtered from {df_state_filtered.index.min().strftime('%Y-%m-%d')} to {df_state_filtered.index.max().strftime('%Y-%m-%d')}.")

# Select the target variable for EDA
geo_data = df_state_filtered['cases_new'].copy()

# Check for missing values in cases_new after filtering
# No missing values should be expected.
missing_values_count = geo_data.isnull().sum()
if missing_values_count > 0:
    print(f"Found {missing_values_count} missing values in after filtering. Forward filling.")
    geo_data = geo_data.ffill()
else:
    print("No missing values found after filtering")

# Aggregate by state weekly intervals
# This is explicitly requested for geospatial analysis to reduce noise and create smoother trends.
df_state_weekly = df_state_filtered.groupby('state')['cases_new'].resample('W').sum().reset_index()
df_state_weekly.rename(columns={'cases_new': 'total_weekly_cases'}, inplace=True)
print("\nState-level data aggregated to weekly totals.")
print(df_state_weekly.head())

# 4.2 REGIONAL DISTRIBUTION & GEOSPATIAL PATTERNS
print("\n========== 4.2 REGIONAL DISTRIBUTION & GEOSPATIAL PATTERNS (State Level) ==========")

# --- Regional Distribution (Overall Average Cases per State) ---
# purpose: observe average cases in each state
avg_cases_per_state = df_state_weekly.groupby('state')['total_weekly_cases'].mean().sort_values(ascending=False)

# plot
plt.figure(figsize=(14, 7))
avg_cases_per_state.plot(kind='bar', color='lightcoral')
plt.title('Average Weekly COVID-19 Cases per State (Jan 2022 onwards)')
plt.xlabel('State')
plt.ylabel('Average Weekly Cases')
plt.xticks(rotation=90)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('state_eda_avg_weekly_cases_per_state.png')
plt.show()

# -- Top 4 States time series plot (Line plot)
# Purpose: Visualize temporal trends for the states with highest average cases.
top_num_states = 4
top_states_names = avg_cases_per_state.head(top_num_states).index

plt.figure(figsize=(16, 8))
# loop the state names extracted from the index
for state_name in top_states_names:
    state_data = df_state_weekly[df_state_weekly['state'] == state_name]
    plt.plot(state_data['date'], state_data['total_weekly_cases'], label=state_name, alpha=0.8)
plt.title(f'Weekly COVID-19 Cases Over Time for Top {top_num_states} States (Jan 2022 onwards)')
plt.xlabel('Date')
plt.ylabel('Weekly New Cases')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('state_eda_top_states_timeseries.png')
plt.show()

# --- Heatmap of Weekly Cases (States vs. Time) ---
# Purpose: Visualize spatio-temporal patterns across states over time, identifying hotspots.
df_state_pivot = df_state_weekly.pivot_table(index='date', columns='state', values='total_weekly_cases')
plt.figure(figsize=(18, 10))
sns.heatmap(df_state_pivot.T, cmap='viridis', cbar_kws={'label': 'Weekly New Cases'})
plt.title('Heatmap of Weekly COVID-19 Cases Across States Over Time (Jan 2022 onwards)')
plt.xlabel('Date')
plt.ylabel('State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('state_eda_heatmap_weekly_cases.png')
plt.show()

# 4.3 SUMMARY ANALYSIS (State Level)
print("\n========== 4.3 SUMMARY ANALYSIS (State Level) ==========")

# --- Numerical Summary for State-level 'cases_new' (Pre-aggregation) ---
print("\n--- Numerical Summary for State-level 'cases_new' (original counts, Jan 2022 onwards) ---")
print(df_state_filtered['cases_new'].describe())

# --- Numerical Summary for Weekly Aggregated State Data ---
print("\n--- Numerical Summary for State-level 'total_weekly_cases' (weekly aggregated, Jan 2022 onwards) ---")
print(df_state_weekly['total_weekly_cases'].describe())

# --- Box Plot of Weekly Aggregated Cases per State (show variability across states) ---
plt.figure(figsize=(14, 7))
sns.boxplot(x='state', y='total_weekly_cases', hue='state', data=df_state_weekly.sort_values
            (by='total_weekly_cases', ascending=False), palette='viridis', legend=False)
plt.title('Distribution of Total Weekly COVID-19 Cases per State (Jan 2022 onwards)')
plt.xlabel('State')
plt.ylabel('Total Weekly Cases')
plt.xticks(rotation=90)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('state_eda_boxplot_weekly_cases_per_state.png')
plt.show()

# ==================================================
# PART 5: ARIMA MODEL DEVELOPMENT (USING CASES_MALAYSIA.CSV)
# ==================================================
print("\n--- Starting PART 5: ARIMA MODEL DEVELOPMENT (USING CASES_MALAYSIA.CSV) ---")

# 5.1 Stationarity Check (ADF Test) on smoothed data for informational purposes
def test_stationarity(timeseries, title):
    print(f"Results of Augmented Dickey-Fuller Test for: {title}")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    if dftest[1] <= 0.05:
        print(f"The time series is stationary (p-value <= 0.05).")
        return 0
    else:
        print(f"The time series is non-stationary (p-value > 0.05) and differencing is likely needed.")
        return 1

# Run ADF test for informational purposes. auto_arima can find 'd' itself
d_order_suggested_from_adf = test_stationarity(ts_data_smoothed, "Smoothed National Data")

# 5.2 ACF/PACF plots of smoothed national data
# Purpose: visualize the temporal dependencies in the smoothed series,
# providing insights for what auto_arima will discover for p, q parameters
# Smoothing specifically helps to reduce the visual impact of the strong weekly seasonality
# allowing the non-seasonal ARIMA to focus on underlying trends and shorter-term autocorrelation.
print("\n--- 5.2 ACF/PACF Plots of Smoothed National Data (for p, q determination) ---")
lags_to_plot = 50
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
plot_acf(ts_data_smoothed, lags=lags_to_plot, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF) - Causal Smoothed Series')
plot_pacf(ts_data_smoothed, lags=lags_to_plot, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF) - Causal Smoothed Series')
plt.tight_layout()
plt.savefig('arima_eda_acf_pacf_smoothed_for_model.png')
plt.show()


# 5.3 AUTOMATIC ARIMA Model Selection using auto_arima
print("\n--- 5.3 AUTOMATIC ARIMA Model Selection using pmdarima.auto_arima ---")

train_size = int(len(ts_data_smoothed) * 0.7)
train = ts_data_smoothed.iloc[:train_size]
test = ts_data_smoothed.iloc[train_size:]

print(f"Train size for auto_arima: {len(train)}, Test size: {len(test)}")
print(f"Train period: {train.index.min().date()} to {train.index.max().date()}")
print(f"Test period: {test.index.min().date()} to {test.index.max().date()}")

# Explain the selection criterion as per your proposal
print("\nSearching for optimal ARIMA parameters (p,d,q)(P,D,Q)m by evaluating multiple configurations")
print("The final model will be selected based on the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) values to balance model fit and complexity")
print("While the dataset shows 'short-term seasonality', this ARIMA model strictly adheres to a non-seasonal (p,d,q) structure as specified in the proposal")
print("The 7-day moving average pre-processing step is important as it aims to mitigate the impact of this seasonality, allowing the non-seasonal ARIMA model to capture underlying trends more effectively")

# Use auto_arima to find the best (p,d,q) order for a pure ARIMA.
# Max orders can be adjusted based on computational time vs. potential for better fit.
max_order_val = 5 # You can experiment with higher values if needed, e.g., 7 or 10
arima_model = pm.auto_arima(train,
                                 start_p=0, start_q=0,
                                 max_p=max_order_val, max_q=max_order_val,
                                 d=None, max_d=2, # Let auto_arima determine differencing
                                 seasonal=False, # Set to false as we are focusing on ARIMA only
                                 trace=True,
                                 suppress_warnings=True,
                                 stepwise=True,
                                 error_action='ignore',
                                 information_criterion='aic') # As stated in your proposal (AIC is the default)

print("\n--- Optimal ARIMA Model Found ---")
print(arima_model.summary())

# Extract the optimal orders
p_order, d_order, q_order = arima_model.order

print(f"\nOptimal Non-Seasonal ARIMA Order (p,d,q): ({p_order},{d_order},{q_order})")

# 5.4 ARIMA Model Training, Forecasting & Diagnostics (using the optimal order)
# Now, fit the statsmodels ARIMA using the order found by auto_arima
# Pass the seasonal_order as well if arima_model found one.
model = ARIMA(train, order=(p_order, d_order, q_order))
model_fit = model.fit()

print(model_fit.summary()) # Print the full summary of the final model

# forecast for the test horizon
n_steps = len(test)
forecast_res = model_fit.get_forecast(steps=n_steps)
predictions = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

# align the predictions
predictions.index = test.index
conf_int.index = test.index

# evaluation metrics implementation
MSE = mean_squared_error(test, predictions)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(test, predictions)

# Check against RMSE target (15% of mean cases in test set)
mean_cases_in_test = test.mean()
rmse_target = 0.15 * mean_cases_in_test
print(f"\nTarget RMSE (15% of mean cases in test set): {rmse_target:.3f}")
if RMSE < rmse_target:
    print(f"Achieved RMSE ({RMSE:.3f}) is below the target ({rmse_target:.3f}).")
else:
    print(f"Achieved RMSE ({RMSE:.3f}) is NOT below the target ({rmse_target:.3f}). This indicates a potential area for improvement or a limitation of the model type for this data.")

print("\n--- ARIMA Evaluation Metrics (using optimal order) ---")
print(f"Order: ARIMA({p_order},{d_order},{q_order}")
print(f"MSE:  {MSE:.3f}")
print(f"RMSE: {RMSE:.3f}")
print(f"MAE:  {MAE:.3f}")

# prediction vs actual plot (using the new optimal order)
plt.figure(figsize=(14,7))
plt.plot(train.index, train, label="Training (7-day MA)", alpha=0.7)
plt.plot(test.index, test, label="Actual Test (7-day MA)", color='black', alpha=0.8)
plt.plot(predictions.index, predictions, label=f"ARIMA({p_order},{d_order},{q_order}) Prediction", color='red', alpha=0.9)
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.25, label='95% CI')
plt.title(f"ARIMA({p_order},{d_order},{q_order}) Forecast vs Actual (Smoothed Cases)")
plt.xlabel("Date")
plt.ylabel("Daily Cases (7-day MA)")
plt.legend()
plt.grid(True)
plt.savefig('arima_forecast_vs_actual.png')
plt.show()

# Residual diagnostics for the new model, essential for verifying model assumptions.
# Ideally, residuals should be white noise (no autocorrelation, mean zero, constant variance).
residuals = model_fit.resid
plt.figure(figsize=(12,4))
plt.plot(residuals)
plt.title(f'Residuals from ARIMA({p_order},{d_order},{q_order})')
plt.grid(True)
plt.tight_layout()
plt.savefig('arima_residuals_timeseries.png')
plt.show()

# residual histogram to check normality of residuals.
plt.figure(figsize=(10,4))
sns.histplot(residuals, bins=40, kde=True)
plt.title('Histogram of ARIMA Residuals')
plt.tight_layout()
plt.savefig('arima_residuals_hist.png')
plt.show()

# QQ plot, for normality of residuals.
plt.figure(figsize=(6,6))
sm.qqplot(residuals, line='s', ax=plt.gca())
plt.title('QQ-plot of ARIMA Residuals')
plt.tight_layout()
plt.savefig('arima_residuals_qq.png')
plt.show()

# Ljung box test for residual autocorrelation
# It is important to check for remaining autocorrelation at various lags,
# particularly given the dataset's inherent seasonality
# non-seasonal ARIMA model might not fully capture, even with smoothing.
# Checking lags at multiples of 7 is crucial to identify if weekly seasonality persists in residuals.
lb_test = acorr_ljungbox(residuals, lags=[7, 14, 21, 28, 35], return_df=True) # Check a few relevant lags
print("\nLjung-Box test on residuals:")
print(lb_test)
print("Ideally, p-values should be > 0.05, indicating no significant autocorrelation in residuals.")
print("If significant autocorrelation remains, especially at seasonal lags (e.g., 7, 14, 21), it highlights a limitation of a purely non-seasonal ARIMA model for this data.")

# storing result for comparison later with lstm
arima_results = {
    'model': f'ARIMA({p_order},{d_order},{q_order})',
    'mse': MSE,
    'rmse': RMSE,
    'mae': MAE
}
print("\nStored ARIMA results for later comparison:", arima_results)

# ==================================
# PART 6: LSTM MODEL DEVELOPMENT
# ==================================

