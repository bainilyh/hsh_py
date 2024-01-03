# pandas连接；merge；join；
# 连接有四种：左链接；右连接；内连接；外连接
# 列连接；使用merge
# 通过how指定连接形式；默认是inner;outer;left;right；通过on指定列名称
import pandas as pd
import numpy as np

df1 = pd.DataFrame({'Key': ['A', 'B'], 'Col1': [20, 30]})
df2 = pd.DataFrame({'Key': ['B', 'C'], 'Col2': ['Cat', 'Dog']})
print(df1.merge(df2, on='Key', how='left'))

# %%
# 当列名称不一致时候,通过left_on和right_on指定
df1 = pd.DataFrame({'Key': ['A', 'B'], 'Col1': [20, 30]})
df2 = pd.DataFrame({'Key2': ['B', 'C'], 'Col2': ['Cat', 'Dog']})
df1.merge(df2, left_on='Key', right_on='Key2', how='left')  # 多出来了Key2列
# %%
# 同名列的区分；通过suffixes添加后缀
# %%
# 多健合并on上添加list
df1 = pd.DataFrame({"Name": ["Alice", "Bob", "Bob", "Tom"],
                    "Class": [1, 1, 2, 2],
                    "Gender": ["Female", "Male", "Male", "Male"]})
df2 = pd.DataFrame({"Name": ["Tim", "Alice", "Bob", "Bob"],
                    "Class": [2, 1, 2, 1],
                    "Grade": [80, 100, 95, 75]})
df1.merge(df2, on=['Name', 'Class'], how='outer')
#     Name  Class  Gender  Grade
# 0  Alice      1  Female  100.0
# 1    Bob      1    Male   75.0
# 2    Bob      2    Male   95.0
# 3    Tom      2    Male    NaN
# 4    Tim      2     NaN   80.0

# 通过validate='1:1'指定左右健的唯一性；还有'1:m'和'm:1'模式；
df1.merge(df2, on=['Name', 'Class'], how='outer', validate='1:1')
# 以下会报错
# pandas.errors.MergeError: Merge keys are not unique in either left or right dataset; not a one-to-one merge
df1.merge(df2, on='Name', how='outer', validate='1:1')

# %%
# 索引连接：join；与merge没有本质区别；没有on
# 先指定某些列为索引，然后使用join来连接。
df1, df2 = df1.set_index(['Name', 'Class']), df2.set_index(['Name', 'Class'])
df1.join(df2, how='outer')

# 左右表存在同名列时；通过lsuffix和rsuffix指定后缀
# join也没有validate参数

# join与merge支持how='cross'；笛卡尔积
# %%
# 方向连接；concat;axis=0/axis=1
df1 = pd.DataFrame({"A": [1,2], "B": [3,4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
df3 = pd.DataFrame({"C": [9, 10], "D": [11, 12]})
#    A  B  A  B   C   D
# 0  1  3  5  7   9  11
# 1  2  4  6  8  10  12
pd.concat([df1, df2, df3], axis=1)

#    A  B
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8
pd.concat([df1, df2], axis=0) #默认是axis=0