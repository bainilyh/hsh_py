from collections import namedtuple
from enum import Enum
import sqlite3
import pandas as pd
import numpy as np
from talib import *

# %%
StockInfo = namedtuple('StockInfo', ['code', 'date', 'candle', 'volume'])
Candle = namedtuple('Candle', ['open', 'close', 'high', 'low', 'candle_line'])
CandleLine = Enum('CandleLine', ['NONE', 'WHITE_BODY', 'BLACK_BODY'])

# %%
con = sqlite3.connect('/Users/bainily/temp/tmp/tushare.nfa')
sql = """select ts_code, trade_date, open, high, low, close, vol, pct_chg, pre_close 
         from daily_stock_info 
         where trade_date >= "20200101"  
         order by ts_code, trade_date asc"""
stock_info = pd.read_sql_query(sql, con)
con.close()


# %%
# # 给蜡烛描色
# my_list = (stock_info['open'] > stock_info['close']).values
# stock_info['candle_line'] = list(map(lambda x: CandleLine.BLACK_BODY.name if x else CandleLine.WHITE_BODY.name
#                                      , my_list))

# 分组统计函数
def f(group_df):
    group_df_pre = group_df[['open', 'close', 'high', 'low']].shift(1)
    condition1 = group_df['open'] < group_df_pre['low']
    condition2 = group_df['close'] > group_df_pre['high']
    condition = condition1 & condition2
    # 前一个蜡烛占后一个蜡烛的比率
    patten_rate = (group_df_pre['high'] - group_df_pre['low']) / (group_df['close'] - group_df['open'])
    series = pd.Series(map(lambda x: 'BULLISH_ENGULFING' if x else 'NONE', condition), name='patten')
    series.index = group_df.index
    trade_date = group_df['trade_date']
    # trade_date.index = range(0, trade_date.shape[0])
    # patten_rate.index = range(0, patten_rate.shape[0])

    # 计算N天后的涨跌幅
    group_df_next = group_df[['open', 'close', 'high', 'low']].shift(-7)
    pct = (group_df_next['close'] - group_df['close']) / group_df['close']
    # pct.index = range(0, pct.shape[0])

    # 计算最近N天的交易波动率｜前3天的7天的交易波动率
    ema_3_vol = EMA(group_df['vol'], 3)
    vol_pre_3 = group_df['vol'].shift(3)
    ema_pre_3_7_vol = EMA(vol_pre_3, 7)
    # 最近的3天交易与3天前的最近7天占比，越高说明最近交易量上来了。
    vol_rat = ema_3_vol / ema_pre_3_7_vol

    # 判断是否是一字板
    one_word_board = pd.Series(np.where(group_df['high'] == group_df['low'], 1, 0))
    one_word_board.index = group_df.index
    num_one_word_7 = SUM(one_word_board, 7)

    # 近7天的涨幅
    increase_rate_7 = (group_df['close'] / group_df['close'].shift(7)) - 1

    # 近30天低点值
    min_7_low_pre = MIN(group_df['low'], 90)
    bool_7_low = pd.Series(np.where(group_df['low'] == min_7_low_pre, 1, 0))
    bool_7_low.index = group_df.index
    df = pd.concat(
        [series, trade_date, patten_rate, pct, vol_rat, one_word_board, num_one_word_7, increase_rate_7, bool_7_low],
        axis=1)
    df.columns = ['patten', 'date', 'patten_rate', 'pct', 'vol_rat', 'one_word_board', 'num_one_word',
                  'increase_rate_7', 'bool_7_low']
    return df


a = stock_info.groupby('ts_code').apply(f)
b = a.droplevel(1).reset_index()
condition1 = b['num_one_word'] == 0
condition2 = b['patten'] == 'BULLISH_ENGULFING'
condition = condition1 & condition2
b = b[condition].dropna()
b = b.sort_values('vol_rat', ascending=False)[['ts_code', 'date', 'patten_rate', 'pct', 'vol_rat', 'increase_rate_7', 'bool_7_low']]
c = b[b['bool_7_low']==1]
# b[40:80]

# b[b['patten'] == 'BULLISH_ENGULFING'].sort_values('pct', ascending=False)

# TODO 1.通过code和日期获取图形
# TODO 2.通过code和日期获取N天后的涨幅 DONNED
