#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:09:39 2023

@author: bainilyhuang
"""

import pandas as pd
import numpy as np
import sqlite3
import time
import datetime
# %%
import talib
from talib import *
# conda install -c conda-forge ta-lib
import mplfinance as mpf

#%%
# 读取sqlite3数据到dataframe
con = sqlite3.connect('/Users/bainilyhuang/Downloads/hshqt/src/main/resources/db/tushare.nfa')
sql = 'select ts_code, trade_date, pct_chg from daily_stock_info where trade_date >= "20200101" \
    and trade_date <= "20200131" order by ts_code, trade_date asc'
stock_info = pd.read_sql_query(sql, con)
con.close()
#%%
df = stock_info.loc[stock_info['ts_code'] == '000002.SZ'][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
df.index = pd.to_datetime(df['Date'])
df.head()
#%%
# 图像处理
mc = mpf.make_marketcolors(up='r', down='g')
s = mpf.make_mpf_style(marketcolors=mc)
mpf.plot(df, type='candle', style=s, figscale=2.5, fill_between=dict(y1=df['Low'].values, y2=df['High'].values))
#%%
# 指标数据计算
def f(prices):
    indicators_name = ['qfq_open', 'qfq_high', 'qfq_low', 'qfq_close', 
                       'ema_350', 'ema_25', 'atr', 'breakup', 'breakdown']
    
    # 前复权数据
    # prices2 = prices.iloc[::-1]
    zdf = prices['close'] / prices['pre_close'] - 1
    fqyz = (1 + zdf).cumprod()
    qfq_close = fqyz * (prices.iloc[-1]['close'] / fqyz.iloc[-1])
    qfq_open = prices['open'] / prices['close'] * qfq_close
    qfq_high = prices['high'] / prices['close'] * qfq_close
    qfq_low = prices['low'] / prices['close'] * qfq_close
    
    # 指标数据
    # prices_close = prices['close']
    ema_350 = EMA(qfq_close, 200)
    ema_25 = EMA(qfq_close, 25)
    atr = ATR(qfq_high, qfq_low, qfq_close, 7)
    breakup = MAX(qfq_close, 20)
    breakdown = MIN(qfq_close, 10)
    # std_dev = STDDEV(prices_close)
    # rsi = RSI(prices_close, 14)
    # ma = MA(prices_close, 5)
    # dif, dem, histogram = MACD(prices_close, 12, 26, 9)
    # histogram *= 2
    # upperband, middleband, lowerband = BBANDS(prices_close, timeperiod= 5)
#     period_max = MAX(prices_close[::-1], 7)
#     period_min = MIN(prices_close[::-1], 7)
#     mfe = abs(period_max - prices_close)
#     mae = abs(period_min - prices_close)
    
    
    
    # natr = NATR(prices_high, prices_low, prices_close, 7)
#     e_sub = (mfe - mae) / atr
    
    # 巨慢
    # mae = prices_close[::-1].rolling(5).apply(MAE_ori)[::-1] / atr
    
    
    
    df = pd.concat([qfq_open, qfq_high, qfq_low, qfq_close, 
                    ema_350, ema_25, atr, breakup, breakdown], axis=1)
    df.columns = indicators_name
    return df

stock_info_new = pd.concat([df, 
                            df.groupby('ts_code').apply(f).droplevel(0)]
                           , axis=1)

condition1 = stock_info_new['trade_date'] >= '20210101'
condition2 = stock_info_new['ts_code'].str.endswith('BJ')
condition = condition1 & ~condition2

new_df = stock_info_new.dropna().loc[condition]
pd.io.sql.to_sql(new_df, name='tqa_info', con=con, if_exists='append', index=False)

