from datetime import datetime

SEED = 42

START = datetime(2009,1,1)
END = datetime(2019,12,31)
DATE_FUTURE = (datetime(2018,1,1), datetime(2019,12,31))
DATE_OOT = (datetime(2017,1,1), datetime(2017,12,31))

LAG_NUM = 4
MAX_NANS_COUNT = 0.05

TARGETCOLUMN = 'Close Price'
