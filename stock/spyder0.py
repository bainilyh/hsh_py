# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 00:19:04 2023

@author: huang
"""
#%%
import pandas as pd
import numpy as np
import sqlite3
import mplfinance as mpf 
import talib
from talib import *
import time
import datetime
from datetime import timedelta
from empyrical import cagr, max_drawdown, sharpe_ratio
#%%
# 读取交易表
con = sqlite3.connect('C:/Users/huang/Downloads/tushare.db')
sql = 'SELECT * FROM trade_vo'
stock_trade = pd.read_sql_query(sql, con)
con.close()
#%%
# 读取指标数据表
con = sqlite3.connect('C:/Users/huang/Downloads/tushare.db')
sql = 'SELECT * FROM tqa_info ORDER BY ts_code, trade_date desc'
stock_info = pd.read_sql_query(sql, con)
con.close()
#%%
# 表链接
df = stock_info.merge(stock_trade, how='inner', left_on=['ts_code', 'trade_date']
                      , right_on=['code', 'hold_date'])
df.index = pd.to_datetime(df['trade_date'])
# 用户衡量指标计算
df['pct_chg'] = df['pct_chg'] / 100
#%%
# 衡量指标函数
def f(stock_info):
    returns = stock_info['pct_chg']
    returns = returns.drop(returns.index[0])
    return cagr(returns), max_drawdown(returns), sharpe_ratio(returns, risk_free=0)
a = df.groupby('mark').apply(f)
b = pd.DataFrame(a.tolist(), index=a.index)
b.columns = ['cagr', 'max_drawdown', 'sharpe_ratio']