#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:45:57 2023

@author: bainilyhuang
"""

import pandas as pd
import numpy as np
import sqlite3
from empyrical import cagr, max_drawdown, sharpe_ratio
import empyrical

# 交易表
con = sqlite3.connect('/Users/bainilyhuang/jupyter/tushare.db')
sql = 'select * from trade_vo'
stock_trade = pd.read_sql_query(sql, con)
con.close()
#%%
# 指标数据表
con = sqlite3.connect('/Users/bainilyhuang/Downloads/hshqt/src/main/resources/db/tushare.nfa')
sql = 'select * from tqa_info order by ts_code,trade_date asc'
stock_info = pd.read_sql_query(sql, con)
con.close()
#%%
# 表连接
df = stock_info.merge(stock_trade, how='inner', left_on=['ts_code', 'trade_date'],
                      right_on=['code', 'hold_date'])
df.index = pd.to_datetime(df['trade_date'])
#%%
# 定义处理函数
def f(info):
    # print(info)
    return cagr(info['pct_chg'])
#%%
con = sqlite3.connect('/Users/bainilyhuang/jupyter/tushare.db')
pd.io.sql.to_sql(stock_info, name='tqa_info', con=con, if_exists='append', index=False)
con.close()
