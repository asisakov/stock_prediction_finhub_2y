# Import libraries

import datetime
import os
import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

import config

SEED = config.SEED

# Configure and create logger

current_dir = os.getcwd()

log_path = os.path.join(current_dir, 'LOGS','data_preparation_info.log')

logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define functions


def exctract_features_from_reports(report, start=datetime.datetime(2009, 1, 1), end=datetime.datetime(2019, 12, 31)):
    """
    Function for data retrieval from dict type to pd.DataFrame
    :param report: dictionary with financial report data for each period
    :param start: first date in observation range, default = 01.01.2009
    :param end: last date in observation range, default = 31.12.2019
    :return: dict with DataFrames, which contains structured information from financial reports
    """
    res = dict()
    for i in range(len(report) - 1, -1, -1):
        if report[i]['quarter'] == 0:
            report[i]['year'] -= 1
            report[i]['quarter'] = 4
        curr_date = pd.to_datetime(f"{report[i]['year']}Q{report[i]['quarter']}")
        if curr_date < start:
            break
        # if curr_date > end:
        #     continue

        buff_info = dict()
        for key in ['bs', 'cf', 'ic']:
            for line in report[i]['report'][key]:
                if not isinstance(line, dict):
                    logger.info(f"[CHECK] For {report[i]['symbol']} at {curr_date}, in {key} no info at line: {line}")
                    continue
                if f"{key}_{line['concept']}_{line['unit']}" not in buff_info.keys():
                    if line['value'] == 'N/A':
                        buff_info[f"{key}_{line['concept']}"] = np.nan
                    elif line['value'] == '':
                        buff_info[f"{key}_{line['concept']}"] = 0
                    else:
                        buff_info[f"{key}_{line['concept']}"] = int(line['value'])

        res[curr_date] = buff_info

    result = pd.DataFrame(res)

    return result.transpose()


def make_target_and_features(dat, targetcolumn='Close Price', num=4):
    """
    Function to add columns with prices for past 'num' observations
    :param dat: DataFrame with data and price info for certain period and ticker
    :param targetcolumn: name of column with price information
    :param num: total number of last observations to append
    :return: DataFrame with additional columns for price lags
    """
    data = dat.copy()
    data = data.sort_index()
    for k in range(num):
        data[f'{targetcolumn}_last_{k+1}Q'] = np.nan
    for i in range(len(data.index)-1,-1,-1):
        for k in range(num):
            data.loc[data.index[i],f'{targetcolumn}_last_{k+1}Q'] = data.loc[data.index[i - (k+1)],targetcolumn]
    return data


def form_big_onlyprices(data, targetcolumn='Close Price', num=4):
    """
    Formation of DataFrame for each industry, which contains only price info for current and last 'num' observations
    :param data: dict of DataFrames with data and price info for certain period and ticker
    :param targetcolumn: name of column with price information
    :param num: total number of last observations to append
    :return: dict of DataFrames with only prices for each industry
    """
    res = pd.DataFrame()
    for ticker in data.keys():
        data[ticker]['ticker'] = ticker
        res = pd.concat([res, data[ticker]], sort=True)
    cols = []
    cols.append('ticker')
    for k in range(num):
        cols.append(f'{targetcolumn}_last_{k+1}Q')
    cols.append(targetcolumn)
    return res[cols]


def form_big_dataset(data):
    """
    Formation of DataFrame for each industry, which contains prices and reports info for current and last 'num' observations
    :param data: dict of DataFrames with data and price info for certain period and ticker
    :return: dict with DataFrames, which contains prices and reports info for each industry
    """
    res = pd.DataFrame()
    for ticker in data.keys():
        data[ticker]['ticker'] = ticker
        res = pd.concat([res, data[ticker]], sort=True)
    return res


# Define global variables

START = config.START
END = config.END
LAG_NUM = config.LAG_NUM
MAX_NANS_COUNT = config.MAX_NANS_COUNT
TARGETCOLUMN = config.TARGETCOLUMN

# Open loaded data

reports_path = os.path.join(current_dir, 'DATA', 'reports.pkl')
prices_path = os.path.join(current_dir, 'DATA', 'prices.pkl')

with open(reports_path, 'rb') as outp:
    reports = pickle.load(outp)


with open(prices_path, 'rb') as outp:
    quarter_prices = pickle.load(outp)

logger.info(f"[INFO] Succefully loaded saved data")

# Make some figures with price dynamics

price_plots_path = os.path.join(current_dir, 'Figures', 'price_plots.pdf')

with PdfPages(price_plots_path) as pdf:
    for key in reports.keys():
        fig = plt.figure(figsize=(16,9))
        for ticker in reports[key]:
            plt.plot(quarter_prices[key][ticker], label=ticker)
            plt.ylabel('Price, $')
            plt.xlabel('Date')
        plt.title(f'Prices in {key.upper()} industry')
        plt.legend()
        pdf.savefig(fig)
        plt.close()

logger.info(f"[INFO] Saved price plots to pdf doc at path: {price_plots_path}")

# If prices seems normal, proceed to Financial Report data extraction

data = dict()
for key in reports.keys():
    data[key] = dict()
    for ticker in reports[key]:
        dat1 = exctract_features_from_reports(reports[key][ticker]['data'])
        dat2 = dat1.join(quarter_prices[key][ticker])
        data[key][ticker] = make_target_and_features(dat2, targetcolumn=config.TARGETCOLUMN, num=config.LAG_NUM)

logger.info("[INFO] Initial data is prepared")

# Check formed dataset on Apple company

check_formed_data_path = os.path.join(current_dir, 'DATA', 'AAPL_full.xlsx')
data['it']['AAPL'].to_excel(check_formed_data_path, index=True)

logger.info(f"[INFO] Saved price plots to pdf doc at path: {price_plots_path}")

# Save initial data for each company

data_path = os.path.join(current_dir, 'DATA', 'data.pkl')

with open(data_path, 'wb') as outp:
    pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

logger.info(f"[INFO] Succefully saved data at {data_path}")

# Make dataset which contains only price info

formed_data_short = dict()

for key in data.keys():
    formed_data_short[key] = form_big_onlyprices(data[key], targetcolumn=config.TARGETCOLUMN, num=config.LAG_NUM)
    formed_data_short[key] = formed_data_short[key].loc[(formed_data_short[key].index >= START) & (formed_data_short[key].index <= END)]

# Save only prices dataset

data_short_path = os.path.join(current_dir, 'DATA', 'data_short_prices.pkl')

with open(data_short_path, 'wb') as outp:
    pickle.dump(formed_data_short, outp, pickle.HIGHEST_PROTOCOL)

logger.info(f"[INFO] Succefully saved data with only prices at {data_short_path}")

# Make dataset with financial data

formed_data_fr = dict()
for key in data.keys():
    formed_data_fr[key] = form_big_dataset(data[key])

# Drop columns with a lot of NaN's

formed_data_short_fr = dict()

dropcols = dict()
for key in data.keys():
    dropcols[key] = []
    rowcount, colcount = formed_data_fr[key].shape[0], formed_data_fr[key].shape[1]

    for column in formed_data_fr[key].columns:
        if sum(formed_data_fr[key][column].isna()) / rowcount > MAX_NANS_COUNT:
            dropcols[key].append(column)

    dropcols[key] = [i for i in dropcols[key] if not i.startswith(config.TARGETCOLUMN)]

    formed_data_short_fr[key] = formed_data_fr[key].drop(columns=dropcols[key])

    formed_data_short_fr[key] = formed_data_short_fr[key].loc[(formed_data_short_fr[key].index >= START) & (formed_data_short_fr[key].index <= END)]

# Save dataset with financial info

data_fr_path = os.path.join(current_dir, 'DATA', 'data_short_financial.pkl')

with open(data_fr_path, 'wb') as outp:
    pickle.dump(formed_data_short_fr, outp, pickle.HIGHEST_PROTOCOL)

logger.info(f"[INFO] Succefully financial data at {data_fr_path}")
