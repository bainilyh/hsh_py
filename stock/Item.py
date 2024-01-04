from collections import namedtuple
from enum import Enum
import sqlite3
import pandas as pd

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
    patten_rate = (group_df_pre['high'] - group_df_pre['low'])/(group_df['close'] - group_df['open'])
    series = pd.Series(map(lambda x: 'BULLISH_ENGULFING' if x else 'NONE', condition), name='patten')
    series.index = range(0, series.shape[0])
    trade_date = group_df['trade_date']
    trade_date.index = range(0, trade_date.shape[0])
    patten_rate.index = range(0, patten_rate.shape[0])

    # 计算N天后的涨跌幅
    group_df_next = group_df[['open', 'close', 'high', 'low']].shift(-7)
    pct = (group_df_next['close'] - group_df['close']) / group_df['close']
    pct.index = range(0, pct.shape[0])
    df = pd.concat([series, group_df['trade_date'], patten_rate, pct], axis=1)
    df.columns = ['patten', 'date', 'patten_rate', 'pct']
    return df

a = stock_info.groupby('ts_code').apply(f)
b = a.droplevel(1).reset_index()

# b[b['patten'] == 'BULLISH_ENGULFING'].sort_values('pct', ascending=False)

# TODO 1.通过code和日期获取图形
# TODO 2.通过code和日期获取N天后的涨幅

