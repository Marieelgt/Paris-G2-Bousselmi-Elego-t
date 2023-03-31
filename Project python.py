 
# -*- coding: utf-8 -*-
 
# Python Project
# French industry analysis:
# Aeronautical industry:Safran (SAF)
# Railway industry: Alstom (ALO)
# Construction industry: Vinci (DG)
# Materials industry: Saint-Gobain (SGO)
 
 
# Import pacakges to use data from yahoo finance
import pandas_datareader.data as pdr
import yfinance as yf
import pandas as pd
yf.pdr_override()
from datetime import datetime
 
   
# Import datas of 4 companies
def get(tickers, startdate, enddate):
    def data(ticker):
        return pdr.get_data_yahoo(ticker, start=startdate, end=enddate)
    datas = map (data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
 
tickers = ['SAF.PA', 'ALO.PA', 'DG.PA', 'SGO.PA']
all_data = get(tickers, datetime(2017, 1, 1), datetime(2022, 1, 1))
 
 
# Moving average on closing prices only
saf = pdr.get_data_yahoo('SAF.PA',
                          start=datetime(2017, 1, 1),  ##(yyyy, dd, mm)
                          end=datetime(2022, 1, 12))     ##(yyyy, dd, mm)
print(saf)
 
alo = pdr.get_data_yahoo('ALO.PA',
                          start=datetime(2017, 1, 1),  ##(yyyy, dd, mm)
                          end=datetime(2022, 1, 12))     ##(yyyy, dd, mm)
print(alo)
 
dg = pdr.get_data_yahoo('DG.PA',
                          start=datetime(2017, 1, 1),  ##(yyyy, dd, mm)
                          end=datetime(2022, 1, 12))     ##(yyyy, dd, mm)
print(dg)
 
sgo = pdr.get_data_yahoo('SGO.PA',
                          start=datetime(2017, 1, 1),  ##(yyyy, dd, mm)
                          end=datetime(2022, 1, 12))     ##(yyyy, dd, mm)
print(sgo)
 
 
# Inspect the Open and Close values at 2017-01-03 and 2022-11-01
print(saf.iloc[[1,1287], [0, 3]])
print(alo.iloc[[1,1287], [0, 3]])
print(dg.iloc[[1,1287], [0, 3]])
print(sgo.iloc[[1,1287], [0, 3]])
 
 
# Inspect our data
# Stock the value in a DataFrame Structure and select a time serie
# Safran
saf.index
saf.columns
saf.info()
saf.tail()
saf.describe()
tsaf = saf['Close'][-10:]
tsaf
 
# Alstom
alo.index
alo.columns
alo.info()
alo.tail()
alo.describe()
talo = alo['Close'][-10:]
talo
 
# Vinci
dg.index
dg.columns
dg.info()
dg.tail()
dg.describe()
tdg = dg['Close'][-10:]
tdg
 
# Saint-Gobain
sgo.index
sgo.columns
sgo.info()
sgo.tail()
sgo.describe()
tsgo = saf['Close'][-10:]
tsgo
 
 
# Label-based indexing and positional indexing
# Inspect the first rows of November-December 2017
print(saf.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2017-12-31')].head())
print(alo.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2017-12-31')].head())
print(dg.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2017-12-31')].head())
print(sgo.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2017-12-31')].head())
 
 
# Inspect the first rows of 2017
print(saf.loc['2017'].head())
print(alo.loc['2017'].head())
print(dg.loc['2017'].head())
print(sgo.loc['2017'].head())
 
 
# Inspect November 2017
print(saf.iloc[214:236])
print(alo.iloc[214:236])
print(dg.iloc[214:236])
print(sgo.iloc[214:236])
 
 
# Plot the closing prices for the companies
# Import the package Matplotlib pyplot
import matplotlib.pyplot as plt
 
# Plot the closing prices for Safran
saf['Close'].plot(grid=True)
alo['Close'].plot(grid=True)
dg['Close'].plot(grid=True)
sgo['Close'].plot(grid=True)
plt.xlabel('Date')
plt.ylabel('Price in €')
plt.title('Closing Prices of Safran, Alstom, Vinci and Saint-Gobain')
plt.show()
 
 
 
# Inspect the quality of our data
# Sample 20 rows
sample = saf.sample(20)
print(sample)
 
# Resample to monthly level
monthly_saf = saf.resample('M').mean()
print(monthly_saf)
 
# Transform daily data in montlhy data
monthly_saf =saf.asfreq("M", method="bfill")
print(monthly_saf)
 
# Deduct closing from opening
# Add a column diff to saf
saf['diff'] = saf.Open - saf.Close
print (saf['diff'])
 
# Delete the new diff column
del saf['diff']
 
 
# Inspect the quality of our data
# Import the package numpy
import numpy as np
 
# Assign Adjusted Close to daily_close
daily_close = saf[['Adj Close']]
 
# Daily returns
daily_pct_change = daily_close.pct_change()
 
# Replace NA values with 0
daily_pct_change.fillna(0, inplace=True)
 
# Inspect daily returns
print(daily_pct_change)
 
# Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)
print(daily_log_returns)
 
 
# Resample saf to business months, take last observation as value
monthly = saf.resample('BM').apply(lambda x: x[-1])
 
# Calculate the monthly percentage change
monthly.pct_change()
 
# Resample saf to quarters, take the mean as value per quarter
quarter = saf.resample("4M").mean()
 
# Calculate the quarterly percentage change
quarter.pct_change()
 
# Daily returns
daily_pct_change = daily_close / daily_close.shift(1) - 1
print(daily_pct_change)
 
 
# Calculate the daily logarithmic returns of a series of financial data stored in a variable
daily_log_returns_shift = np.log(daily_close / daily_close.shift(1))
print(daily_log_returns_shift)
 
 
# Pull up summary statistics
print(daily_pct_change.describe())
 
# Data Viz
# Calculate the cumulative daily returns
cum_daily_return = (1 + daily_pct_change).cumprod()
print(cum_daily_return)
cum_daily_return.plot(figsize=(12,8))
plt.title('Cumulative daily returns of Safran')
 
# Same thing for the 3 other companies
# Alstom
daily_close = alo[['Adj Close']]
daily_pct_change = daily_close.pct_change()
daily_pct_change.fillna(0, inplace=True)
print(daily_pct_change)
daily_log_returns = np.log(daily_close.pct_change()+1)
print(daily_log_returns)
monthly = alo.resample('BM').apply(lambda x: x[-1])
monthly.pct_change()
quarter = alo.resample("4M").mean()
quarter.pct_change()
daily_pct_change = daily_close / daily_close.shift(1) - 1
print(daily_pct_change)
daily_log_returns_shift = np.log(daily_close / daily_close.shift(1))
print(daily_log_returns_shift)
print(daily_pct_change.describe())
cum_daily_return = (1 + daily_pct_change).cumprod()
print(cum_daily_return)
cum_daily_return.plot(figsize=(12,8))
plt.title('Cumulative daily returns of Alstom')
 
# Vinci
daily_close = dg[['Adj Close']]
daily_pct_change = daily_close.pct_change()
daily_pct_change.fillna(0, inplace=True)
print(daily_pct_change)
daily_log_returns = np.log(daily_close.pct_change()+1)
print(daily_log_returns)
monthly = dg.resample('BM').apply(lambda x: x[-1])
monthly.pct_change()
quarter = dg.resample("4M").mean()
quarter.pct_change()
daily_pct_change = daily_close / daily_close.shift(1) - 1
print(daily_pct_change)
daily_log_returns_shift = np.log(daily_close / daily_close.shift(1))
print(daily_log_returns_shift)
print(daily_pct_change.describe())
cum_daily_return = (1 + daily_pct_change).cumprod()
print(cum_daily_return)
cum_daily_return.plot(figsize=(12,8))
plt.title('Cumulative daily returns of Vinci')
 
# Saint-Gobain
daily_close = sgo[['Adj Close']]
daily_pct_change = daily_close.pct_change()
daily_pct_change.fillna(0, inplace=True)
print(daily_pct_change)
daily_log_returns = np.log(daily_close.pct_change()+1)
print(daily_log_returns)
monthly = sgo.resample('BM').apply(lambda x: x[-1])
monthly.pct_change()
quarter = sgo.resample("4M").mean()
quarter.pct_change()
daily_pct_change = daily_close / daily_close.shift(1) - 1
print(daily_pct_change)
daily_log_returns_shift = np.log(daily_close / daily_close.shift(1))
print(daily_log_returns_shift)
print(daily_pct_change.describe())
cum_daily_return = (1 + daily_pct_change).cumprod()
print(cum_daily_return)
cum_daily_return.plot(figsize=(12,8))
plt.title('Cumulative daily returns of Saint-Gobain')
 
 
# Frequency Distributions on closing prices only
# Isolate the Adjusted Close values and transform the DataFrame
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')
 
# Calculate the daily percentage change for daily_close_px
daily_pct_change = daily_close_px.pct_change()
 
# Summary statistics and cumulative returns
# Plot the Frequency Distributions on closing prices
daily_pct_change.hist(bins=50, sharex=False, figsize=(12,8))
plt.show
 
# Isolate the adjusted closing prices
adj_close_px = saf[['Adj Close']]
 
# Calculate the moving average for saf
moving_avg = adj_close_px.rolling(window=40).mean()
 
# Inspect the result
print(moving_avg[-10:])
 
# Safran
# Short and long moving windows: rolling means
# Short moving window rolling mean
saf['31'] = adj_close_px.rolling(window=31).mean()
 
# Long moving window rolling mean
saf['365'] = adj_close_px.rolling(window=365).mean()
 
# Plot the adjusted closing price, the short and long windows of rolling means
saf[['Adj Close', '31', '365']].plot()
plt.title('Rolling means of Safran')
plt.show()
 
# Alstom
alo['31'] = adj_close_px.rolling(window=31).mean()
alo['365'] = adj_close_px.rolling(window=365).mean()
alo[['Adj Close', '31', '365']].plot()
plt.title('Rolling means of Alstom')
plt.show()
 
# Vinci
dg['31'] = adj_close_px.rolling(window=31).mean()
dg['365'] = adj_close_px.rolling(window=365).mean()
dg[['Adj Close', '31', '365']].plot()
plt.title('Rolling means of Vinci')
plt.show()
 
# Saint-Gobain
sgo['31'] = adj_close_px.rolling(window=31).mean()
sgo['365'] = adj_close_px.rolling(window=365).mean()
sgo[['Adj Close', '31', '365']].plot()
plt.title('Rolling means of Saint-Gobain')
plt.show()
 
# Volatility calculation
# Define the minumum of periods to consider
min_periods = 100
 
# Calculate the volatility
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
 
# Plot the volatility
vol.plot(figsize=(8, 6))
plt.xlabel('Date')
plt.ylabel('Volatility in %')
plt.title('Volatility Calculation')
plt.show()
 
 
# OLS regression
# Import the package api model of statsmodels
import statsmodels.api as sm
# Isolate the adjusted closing price
all_adj_close = all_data[['Adj Close']]
# Calculate the returns
all_returns = np.log(all_adj_close / all_adj_close.shift(1))
# Isolate the Safran returns
saf_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'SAF.PA']
# to drop index "Ticker"
saf_returns.index = saf_returns.index.droplevel('Ticker')
# Isolate the Alstom returns
alo_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'ALO.PA']
alo_returns.index = alo_returns.index.droplevel('Ticker')
 
 
# Build up a new DataFrame with Safran and Alstom returns
return_data = pd.concat([saf_returns, alo_returns], axis=1)[1:]
return_data.columns = ['SAF.PA', 'ALO.PA']
# Add a constant
X = sm.add_constant(return_data['SAF.PA'])
 
# Construct the model
model = sm.OLS(return_data['ALO.PA'],X).fit()
# Print the summary
print(model.summary())
 
# Plotting the OLS Regression
# Plot returns of Safran and Alstom
plt.plot(return_data['SAF.PA'], return_data['ALO.PA'], 'r.')
 
# Add an axis to the plot
ax = plt.axis()
 
# Initialize x
x = np.linspace(ax[0], ax[1] + 0.01)
# x will help me to plot OLS regression // here x varies between min and max+0.01
 
# Plot the regression line
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)
# my OLS regression line : y = 0.0006 + 0.3791 * x
 
# Customize the plot
plt.grid(True)
plt.axis('tight')
plt.xlabel('Safran Returns')
plt.ylabel('Alstom returns')
 
# Show the plot
plt.title('OLS Regression')
plt.show()
 
# Now that we have done the OLS Regression for Safran and Alstom
# We are going to do it for Vinci and Saint-Gobain
all_adj_close = all_data[['Adj Close']]
# Calculate the returns
all_returns = np.log(all_adj_close / all_adj_close.shift(1))
# Isolate the Vinci returns
dg_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'DG.PA']
dg_returns.index = dg_returns.index.droplevel('Ticker')
 
# Isolate the Saint-Gobain returns
sgo_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'SGO.PA']
sgo_returns.index = sgo_returns.index.droplevel('Ticker')
 
# Build up a new DataFrame with Vinci and Saint-Gobain returns
return_data = pd.concat([dg_returns, sgo_returns], axis=1)[1:]
return_data.columns = ['DG.PA', 'SGO.PA']
# Add a constant
X = sm.add_constant(return_data['DG.PA'])
 
# Construct the model
model = sm.OLS(return_data['SGO.PA'],X).fit()
# Print the summary
print(model.summary())
 
# Plotting the OLS Regression
# Plot returns of Vinci and Saint-Gobain
plt.plot(return_data['DG.PA'], return_data['SGO.PA'], 'r.')
 
# Add an axis to the plot
ax = plt.axis()
 
# Initialize x
x = np.linspace(ax[0], ax[1] + 0.01)
 
# Plot the regression line
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)
 
 
# Customize the plot
plt.grid(True)
plt.axis('tight')  # axis are just large enough to show all data
plt.xlabel('Vinci Returns')
plt.ylabel('Saint-Gobain returns')
plt.title('OLS Regression')
plt.show()
 
 
# Creating signals with the Simple Moving Averages strategy (SMA)
# Initialize the short and long windows
short_window = 31
long_window = 365
 
# Initialize the signals DataFrame with the signal column
signals = pd.DataFrame(index=saf.index)
signals['signal'] = 0.0
 
# Create short simple moving average over the short window
signals['short_mavg'] = saf['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
 
# Create long simple moving average over the long window
signals['long_mavg'] = saf['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
 
# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)  
 
# Generate trading orders
signals['positions'] = signals['signal'].diff()
print(signals)
 
 
# Plot our signals
# Initialize the plot figure
fig = plt.figure()
 
# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in €')
 
# Plot the closing price
saf['Close'].plot(ax=ax1, color='m', lw=2.)
 
# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
 
# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='k')
        
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='r')
        
# Show the plot
plt.title('Plot signals')
plt.show()
 
 
# Backtesting a trading strategy
# Set the initial capital
initial_capital= float(100000.0)
 
# Create a DataFrame positions
positions = pd.DataFrame(index=signals.index).fillna(0.0)
 
# Buy 100 shares
positions['SAF'] = 100*signals['signal']  
  
# Initialize the portfolio with value owned  
portfolio = positions.multiply(saf['Adj Close'], axis=0)
 
# Store the difference in shares owned
pos_diff = positions.diff()
 
# Add holdings to portfolio
portfolio['holdings'] = (positions.multiply(saf['Adj Close'], axis=0)).sum(axis=1)
 
# Add cash to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(saf['Adj Close'], axis=0)).sum(axis=1).cumsum()  
 
# Add total to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
 
# Add returns to portfolio
portfolio['returns'] = portfolio['total'].pct_change()
 
# Print the first lines of portfolio
print(portfolio.head())
 
 
# Visualize our portfolio
# Create a figure
fig = plt.figure()
 
ax1 = fig.add_subplot(111, ylabel='Portfolio value in €')
 
# Plot the equity curve in euros
portfolio['total'].plot(ax=ax1, lw=2.)
 
ax1.plot(portfolio.loc[signals.positions == 1.0].index,
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='r')
ax1.plot(portfolio.loc[signals.positions == -1.0].index,
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='g')
 
# Show the plot
plt.title('Visualization of the portfolio')
plt.show()
 
 
# Evaluating Moving Average Crossover Strategy
# Isolate the returns of our strategy
returns = portfolio['returns']
 
# Annualized Sharpe ratio
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
 
print(sharpe_ratio)
 
 
# Maximum drawdown
# Define a trailing 252 trading day window
window = 252
 
# Calculate the max drawdown in the past window days for each day
rolling_max = saf['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = saf['Adj Close']/rolling_max - 1.0
 
# Calculate the minimum (negative) daily drawdown
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
 
# Plot the results
daily_drawdown.plot()
max_daily_drawdown.plot()
 
# Show the plot
plt.title('Maximum drawdown')
plt.show()
 
 
# Compound Annual Growth Rate (CAGR)
# Get the number of days
# Get the number of days in Safran
days = (saf.index[-1] - saf.index[0]).days
 
# Calculate the CAGR
cagr = ((((saf['Adj Close'][-1]) / saf['Adj Close'][1])) ** (365.0/days)) - 1
print(cagr)
 
cagr = ((((alo['Adj Close'][-1]) / alo['Adj Close'][1])) ** (365.0/days)) - 1
print(cagr)
 
cagr = ((((dg['Adj Close'][-1]) / dg['Adj Close'][1])) ** (365.0/days)) - 1
print(cagr)
 
cagr = ((((sgo['Adj Close'][-1]) / sgo['Adj Close'][1])) ** (365.0/days)) - 1
print(cagr)
 
 
# With this code we can compare, the health of 4 companies of each industry
# in comparison with the others
# We also did a quick portfolio simulation
 
