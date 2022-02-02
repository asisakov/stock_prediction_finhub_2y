# Import libraries

import datetime
import os
import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error#, mean_absolute_percentage_error

from lightgbm import LGBMRegressor
import lightgbm as lgb

import config

SEED = config.SEED

# Configure and create logger

current_dir = os.getcwd()

log_path = os.path.join(current_dir, 'LOGS','modeling_info.log')

logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define functions


def form_data_and_models(dat, date_future=(datetime.datetime(2018,1,1), datetime.datetime(2019,12,31)),
                         date_oot=(datetime.datetime(2017,1,1), datetime.datetime(2017,12,31))):

    prepared_short = dict()
    models = dict()

    for k in ['RF', 'LR', 'GB']:
        models[k] = dict()
    for key in dat.keys():
        prepared_short[key] = dat[key].dropna()
        models['RF'][key] = RandomForestRegressor(random_state=SEED)
        models['LR'][key] = LinearRegression()
        models['GB'][key] = LGBMRegressor(random_state=SEED)

    future = dict()
    oot = dict()
    dev = dict()
    for key in dat.keys():
        dev[key] = prepared_short[key].loc[prepared_short[key].index < date_oot[0]]
        oot[key] = prepared_short[key].loc[(prepared_short[key].index >= date_oot[0]) & (prepared_short[key].index <= date_oot[1])]
        future[key] = prepared_short[key].loc[(prepared_short[key].index >= date_future[0]) & (prepared_short[key].index <= date_future[1])]

    return models, dev, oot, future

# Define global variables

START = config.START
END = config.END
LAG_NUM = config.LAG_NUM
MAX_NANS_COUNT = config.MAX_NANS_COUNT
TARGETCOLUMN = config.TARGETCOLUMN

DATE_FUTURE = config.DATE_FUTURE
DATE_OOT = config.DATE_OOT

# Open loaded data

path_data_financial = os.path.join(current_dir, 'DATA', 'data_short_financial.pkl')
path_data_prices = os.path.join(current_dir, 'DATA', 'data_short_prices.pkl')

data = dict()

with open(path_data_prices, 'rb') as outp:
    data['only_prices'] = pickle.load(outp)

with open(path_data_financial, 'rb') as outp:
    data['financial'] = pickle.load(outp)

logger.info(f"[INFO] Succefully loaded saved data")

# Make modeling without specific data preparation
# Start from dictionaries with data and models for each industry

for fin_key in data.keys():

    logger.info(f"[INFO] Start working with {fin_key.upper()} data")

    models, dev, oot, future = form_data_and_models(data[fin_key], date_future=DATE_FUTURE, date_oot=DATE_OOT)
    for key in data[fin_key].keys():
        logger.info(f"For current key={key}, dev shape is {dev[key].shape}, oot shape is {oot[key].shape}")

    # Check one of the formed datasets

    check_formed_set_path = os.path.join(current_dir, 'Output', f'dev_set_{fin_key}_IT.xlsx')
    dev['it'].to_excel(check_formed_set_path, index=True)

    logger.info(f"[INFO] Saved data for IT to check at path: {check_formed_set_path}")

    # Train/test split

    dev_set = dict()
    for key in dev.keys():
        dev_set[key] = dict()
        dev_set[key]['X_train'], dev_set[key]['X_test'], dev_set[key]['y_train'], dev_set[key]['y_test'] = train_test_split(
            dev[key].drop(columns=[TARGETCOLUMN, 'ticker']), dev[key]['Close Price'], test_size=0.3, random_state=SEED)

    X_train, X_test, y_train, y_test = dev_set[key]['X_train'], dev_set[key]['X_test'], dev_set[key]['y_train'], dev_set[key]['y_test']

    scores_fr = dict()
    for key in tickers.keys():
        scores_fr[key] = dict()
        for typ in models_fr.keys():
            scores_fr[key][typ] = dict()
            models_fr[typ][key].fit(data_fr[key]['X_train'], data_fr[key]['y_train'])

            scores_fr[key][typ]['R2_train'] = r2_score(data_fr[key]['y_train'],
                                                       models_fr[typ][key].predict(data_fr[key]['X_train']))
            scores_fr[key][typ]['MSE_train'] = mean_squared_error(data_fr[key]['y_train'],
                                                                  models_fr[typ][key].predict(data_fr[key]['X_train']),
                                                                  squared=True)
            scores_fr[key][typ]['MAPE_train'] = mean_absolute_percentage_error(data_fr[key]['y_train'],
                                                                               models_fr[typ][key].predict(
                                                                                   data_fr[key]['X_train']))

            scores_fr[key][typ]['R2_test'] = r2_score(data_fr[key]['y_test'],
                                                      models_fr[typ][key].predict(data_fr[key]['X_test']))
            scores_fr[key][typ]['MSE_test'] = mean_squared_error(data_fr[key]['y_test'],
                                                                 models_fr[typ][key].predict(data_fr[key]['X_test']),
                                                                 squared=True)
            scores_fr[key][typ]['MAPE_test'] = mean_absolute_percentage_error(data_fr[key]['y_test'],
                                                                              models_fr[typ][key].predict(
                                                                                  data_fr[key]['X_test']))

            models_fr[typ][key].fit(dev_fr[key].drop(columns=['Close Price', 'ticker']), dev_fr[key]['Close Price'])

            scores_fr[key][typ]['R2_oot'] = r2_score(oot_fr[key]['Close Price'], models_fr[typ][key].predict(
                oot_fr[key].drop(columns=['Close Price', 'ticker'])))
            scores_fr[key][typ]['MSE_oot'] = mean_squared_error(oot_fr[key]['Close Price'], models_fr[typ][key].predict(
                oot_fr[key].drop(columns=['Close Price', 'ticker'])), squared=True)
            scores_fr[key][typ]['MAPE_oot'] = mean_absolute_percentage_error(oot_fr[key]['Close Price'],
                                                                             models_fr[typ][key].predict(
                                                                                 oot_fr[key].drop(
                                                                                     columns=['Close Price',
                                                                                              'ticker'])))




    break


