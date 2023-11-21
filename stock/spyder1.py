# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 11:19:13 2023

@author: huang
"""

import pandas as pd
import numpy as np
import sqlite3
import mplfinance as mpf 
from datetime import datetime
from datetime import timedelta
#%%
# 读取指标数据表
con = sqlite3.connect('C:/Users/huang/Downloads/tushare.db')
sql = 'SELECT * FROM tqa_info ORDER BY ts_code, trade_date desc'
stock_info = pd.read_sql_query(sql, con)
con.close()
#%%
df = stock_info[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']]
df.columns = ['code', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df.index = pd.to_datetime(df['Date'])
df = df[::-1]
#%%
# 筛选合适的区间范围1
start_time = datetime(2022, 3, 16) - timedelta(30)
end_time = datetime(2022, 4, 15) + timedelta(30)
condition1 = df.index >= start_time
condition2 = df.index <= end_time
condition3 = df['code'] == '000965.SZ'
condition = condition1 & condition2 & condition3
daily = df.loc[condition]
#%%
# 筛选合适的区间范围2
start_time = c[0] - timedelta(30)
end_time = c[-1] + timedelta(30)
condition1 = df.index >= start_time
condition2 = df.index <= end_time
condition3 = df['code'] == '002150.SZ'
condition = condition1 & condition2 & condition3
daily = df.loc[condition]
#%%

# 设置标记
def test_scatter(df, dates):
    signal = [df.loc[date]['High'] if date in dates else np.nan for date in df.index]
    return signal
dates = set([datetime(2022, 3, 16), datetime(2022, 4, 15)])
signal = test_scatter(daily, dates)

# show_nontrading=True 显示非交易日
# title='\nS&P 500, Nov 2019' 设置标题
# ylabel='OHLC Candles' 设置y轴标题
# ylabel_lower='Shares\nTraded' 设置y轴volume标题
# xlabel='DATE' 设置x轴标题
# xrotation=20 设置x轴标签旋转
# datetime_format=' %A, %d-%m-%Y' 设置日期格式
# linecolor='#00ff00' 当type='line'设置line颜色
# tight_layout=True 设置紧凑图片
# mav=6 mav=(3,6)

# [fill_between 设置填充;1.可以是一个值/数组等，默认是y1=0 y2=值之间的填充
# 或者自己设置y1和y2
# fill_between=dict(y1=daily['Low'].values, y2=daily['High'].values, color='g')
# where 是一个与数据帧长度相同的布尔系列。]

kwargs = dict(type='candle',volume=True,figscale=1.5, figratio=(28, 16)
              , datetime_format='%d-%m-%Y'
              , volume_panel=1
              # , title='000001.SZ'
              )
# 设置自己的风格
# marketcolors必须设置；mavcolors不必须；matplotlib style不必须
mc = mpf.make_marketcolors(up='red', down='green', edge='black', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc)
close_plot = mpf.make_addplot(daily['Close'], color='black')
apd = mpf.make_addplot(signal, type='scatter', markersize=200, marker='v')
# open_plot = mpf.make_addplot(daily['Open'], color='red', panel=1)
plots = [close_plot, apd]
mpf.plot(daily, **kwargs, style=s, addplot=plots)
#%%

