#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

import pandas as pd
import numpy as np
import sqlite3
import time
import datetime
import pickle
# %%
con = sqlite3.connect('/Users/bainilyhuang/Downloads/hshqt/src/main/resources/db/tushare.nfa')
sql = 'select ts_code, trade_date, pct_chg from daily_stock_info where trade_date >= "20000101"  order by ts_code, trade_date asc'
stock_info = pd.read_sql_query(sql, con)
con.close()
# %%
from collections import defaultdict
code_chg_list = defaultdict(list)
for ts_code, trade_date, pct_chg in stock_info.values:
    chg_list = code_chg_list.get(ts_code, [])
    chg_list.append(pct_chg)
    code_chg_list[ts_code] = chg_list

# 节约内存
del stock_info

data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 将数据保存到pct_plk
with open(os.path.join(data_dir, 'pct_chg.plk'), 'wb') as f:
    pickle.dump(code_chg_list, f)

# %%

with open('./data/pct_chg.plk', 'rb') as f:
    code_chg_list = pickle.load(f)

print(len(code_chg_list))
# 采样分割


