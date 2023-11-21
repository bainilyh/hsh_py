# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:53:30 2023

@author: huang
"""
import sqlite3
import pandas as pd
import collections
from collections import namedtuple
from collections import defaultdict
#%%
con = sqlite3.connect('/Users/bainilyhuang/jupyter/tushare.db')
sql = 'SELECT * FROM tqa_info ORDER BY ts_code, trade_date desc'
stock_info = pd.read_sql_query(sql, con)
con.close()
#%%


Stock = namedtuple('stock_info', ['code', 'date', 'open'
                                              , 'close', 'high', 'low'
                                              , 'volume', 'slow_ma'
                                              , 'quick_ma', 'atr', 'breakup'
                                              , 'breakdown'])

Trade = namedtuple('trade', ['code', 'date', 'mark', 'trade_type'])
# stock = Stock('11', '2032-392-23', 12.0, 231.9, 23, 24, 109313, 23, 42, 423, 423, 243)


stock_list = iter(stock_info[['ts_code', 'trade_date', 'qfq_open'
                         , 'qfq_close', 'qfq_high', 'qfq_low'
                         , 'vol', 'ema_350', 'ema_25', 'atr'
                         , 'breakup', 'breakdown']].values)

stock_info_list = (Stock(*stock) for stock in stock_list)
#%%
stock_map = defaultdict(list)
for info in stock_info_list:
    stock_map[info.code].append(info)
#%%
# 仿写java
k = 0
for key, value in stock_map.items():
    value = value[::-1]
    i = 1
    while i < len(value):
        pre = value[i - 1]
        cur = value[i]
        if pre.quick_ma > pre.slow_ma and cur.close > pre.breakup:
            for j in range(i + 1, len(value)):
                f_pre = value[j - 1]
                f_cur = value[j]
                if f_cur.close < (cur.close - 2 * cur.atr):
                    i = j
                    k += 1
                    break
                if f_cur.close < f_pre.breakdown:
                    i = j
                    k += 1
                    break
        i += 1