# Import libraries

import datetime
import time
import os
import logging

import pandas as pd
import finnhub
import pickle

import config

SEED = 42

# Configure and create logger

current_dir = os.getcwd()

log_path = os.path.join(current_dir, 'LOGS','data_loading_info.log')

logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define functions


def extract_all_reports_from_ticker(ticker_name):
    """
    Function for all reports data extraction from used ticker name
    :param ticker_name: ticker name, for example 'AAPL'
    :return: dict with quarter and annual reports
    """
    quarterly = finnhub_client.financials_reported(symbol=ticker_name, freq='quarterly')
    time.sleep(1)
    annual = finnhub_client.financials_reported(symbol=ticker_name, freq='annual')
    time.sleep(1)
    quarterly['data'].extend(annual['data'])
    return quarterly


def extract_price_from_ticker(ticker_name, start=datetime.datetime(2009,1,1), end=datetime.datetime(2019,12,31), freq='M'):
    """
    Function for prices extraction for defined period and frequency from ticker info
    :param ticker_name: ticker name, for example 'AAPL'
    :param start: first date in observation range, default = 01.01.2009
    :param end: last date in observation range, default = 31.12.2019
    :param freq: price frequency, default is 'M' [1, 5, 15, 30, 60, 'D', 'W', 'M']
    :return: DataFrame, which contains prices with used frequency for used ticker
    """
    if start is None or start == 'None':
        start = datetime.datetime(2009, 1, 1)
    if end is None or end == 'None':
        end = datetime.datetime(2019,12,31)
    if freq is None or freq == 'None':
        freq = 'M'

    # prices = finnhub_client.stock_candles(ticker_name, freq, int(start.timestamp()), int(end.timestamp()))['c']
    prices = finnhub_client.stock_candles(ticker_name, freq, int((start - datetime.datetime(1970,1,1)).total_seconds()),
                                          int((end - datetime.datetime(1970,1,1)).total_seconds()))['c']
    months = pd.date_range(start,end, freq='MS').strftime("%Y-%b")
    res = pd.DataFrame(prices, index=months[-len(prices):], columns=['Close Price'])
    return res


def modify_monthly_price_to_quarter(ticker_price):
    """
    Reformat monthly prices to quarterly frequency by taking the mean value
    :param ticker_price: DataFrame with monthly prices
    :return: DataFrame with quarterly prices
    """
    res = ticker_price.groupby(pd.PeriodIndex(ticker_price.index, freq='Q'))['Close Price'].mean()
    res.index = res.index.to_timestamp()
    return res


# Define global variables

START = config.START
END = config.END

# Connect to finhub API

with open('API_KEY.txt', 'r') as f:
    API_KEY = f.readline()

finnhub_client = finnhub.Client(api_key=API_KEY)
logger.info("[INFO] Connected to finnhub API")

# Open text file with ticker lists

tickers = dict()

with open('INPUT.txt', 'r') as f:
    keys = f.readline()[:-1].split(',')
    for key in keys:
        tickers[key] = f.readline().replace('\n', '').split(',')

# Load reports information

reports = dict()
for key in tickers.keys():
    reports[key] = dict()
    for ticker in tickers[key]:
        reports[key][ticker] = extract_all_reports_from_ticker(ticker)

    logger.info(f"[INFO] Succefully loaded reports for {key} industry")

logger.info(f"[INFO] Number of reports for AAPL in it industry: {len(reports['it']['AAPL']['data'])}")

# Load prices information

quarter_prices = dict()
for key in tickers.keys():
    quarter_prices[key] = dict()
    for ticker in tickers[key]:
        price = extract_price_from_ticker(ticker, start=START, end=END)
        quarter_prices[key][ticker] = modify_monthly_price_to_quarter(price)
        time.sleep(1)
    logger.info(f"[INFO] Succefully loaded quarterly prices for {key} industry")

# Save prices and reports info to pickle format

reports_path = os.path.join(current_dir, 'DATA','reports.pkl')
prices_path = os.path.join(current_dir, 'DATA','prices.pkl')


with open(reports_path, 'wb') as outp:
    pickle.dump(reports, outp, pickle.HIGHEST_PROTOCOL)

logger.info(f"[INFO] Succefully saved reports at {reports_path}")

with open(prices_path, 'wb') as outp:
    pickle.dump(quarter_prices, outp, pickle.HIGHEST_PROTOCOL)

logger.info(f"[INFO] Succefully saved prices at {prices_path}")
