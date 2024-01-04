# 0.pandas基本函数
# df.info() df.describe()
# %%
# pandas连接；merge；join；
# 连接有四种：左链接；右连接；内连接；外连接
# 列连接；使用merge
# 通过how指定连接形式；默认是inner;outer;left;right；通过on指定列名称
import pandas as pd
import numpy as np

# %%
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
# 其他连接：concat()、assign()、compare()和combine()
# 方向连接；concat;axis=0/axis=1
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
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
pd.concat([df1, df2], axis=0)  # 默认是axis=0；同名列会上下拼接；不同名列会左右拼接。

# DataFrame与Series的合并以及Series与Series的合并均可通过concat()函数实现
s = pd.Series([5, 6], name="C")
pd.concat([df1, s], axis=1)  # DataFrame与Series

s = pd.Series([5, 6], index=["A", "B"], name=2)
pd.concat([df1, s.to_frame().T], axis=0)  # 先转为DataFrame再concat

s1 = pd.Series([1, 2], index=["a", "b"], name="Apple")
s2 = pd.Series([3, 4], index=["c", "d"], name="Apple")
s3 = pd.Series([3, 4], index=["a", "b"], name="Banana")
pd.concat([s1, s2], axis=0)  # Series与Series
pd.concat([s1, s3], axis=1)  # Series与Series

# concat两个重要参数join与keys
# join与merge()和join()中的how作用一样；值只有outer/inner
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["label1", "label2"])
df2 = pd.DataFrame({"C": [5, 6]}, index=["label1", "label3"])
df3 = pd.DataFrame({"D": [7, 8]}, index=["label1", "label4"])
pd.concat([df1, df2, df3], axis=1, join="inner")  # 只有合并同名索引的行
pd.concat([df1, df2, df3], axis=1, join="outer")  # 全部合并

# keys TODO （不懂）

# 方向连接只保证索引（即纵向连接的列索引和横向连接的行索引）是一致的。
df_a = pd.DataFrame({"语文": [80, 95, 70], "英语": [90, 92, 80]},
                    index=["张三", "李四", "王五"])
df_b = pd.DataFrame({"数学": [85, 75, 75]}, index=["李四", "张三", "王五"])
pd.concat([df_a, df_b], axis=1)

# 索引左右两个dataframe索引不唯一，方向连接直接报错；关系连接会产生笛卡尔积的拼接结果。
# 多重索引，只有当左右两个dataframe索引一一对应的时候，方向连接才不会报错。
df_b.index = df_a.index
pd.concat([df_a, df_b], axis=1)

# assign()函数把Series加到DataFrame的末列
# 传入的参数名称为新列名
# 支持一次拼接多个序列
# 如果被拼接的Series索引中出现了DataFrame行索引中未出现的元素，拼接结果的相应位置会被设置为缺失值
# 用df["new_col"]=s增加新列，但这样做会对原df做出改动
s1 = pd.Series([5, 6], index=["label1", "label2"])
s2 = pd.Series([5, 6], index=["label1", "label3"])
# 最后一列中整数5被转化为浮点5.0的原因将在第7章中解释
df1.assign(C=s1, D=s2)

# compare比较两个dataframe的不同之处；并列出不同之处的数据；
# combine组合两个dataframe

# %%
# 3.索引
# 单级索引
# dataframe列索引通过df['列名']获取数据，返回的是Series
import os
import pandas as pd

parent_dir = '/Users/bainilyhuang/Downloads/joyful-pandas-master/'
df = pd.read_csv(os.path.join(parent_dir, './data/learn_pandas.csv'),
                 usecols=['School', 'Grade', 'Name', 'Gender', 'Weight', 'Transfer'])
df['Name'].head()  # 单列
df[['Name', 'Gender']].head()  # 多列
# 也可以通过df.列名取出数据，前提是列名不含空格。

# %%
# Series的行索引；通过s['行名(索引名称)']
s = pd.Series([1, 2, 3, 4, 5, 6], index=['a', 'b', 'a', 'a', 'a', 'c'])
s['a']  # 多个值，返回Series
s['b']  # 单值，返回标量
s[['a', 'b']]  # 取多个索引
# 可以通过切片来取多个索引数据s['索引1': '索引2': N]；切片包含两端
# 如果索引出现重复，必须先排序后再通过切片取多个索引
s.sort_index()['a':'b']

# 以整数为索引的Series
s[1:-1].head()  # s[1:-1:2].head()
# 不应该用浮点数或者多个类型混合作为索引

# %%
# dataframe取行索引数据：loc基于元素/iloc基于位置
# loc[*,*] 第一个*选取行索引，第二个*选取列索引
# loc[*] 选取某个行索引，以及所有的列索引数据
# *，可以由元素、元素列表、切片、布尔列表表示。
df_demo = df.set_index('Name')  # 将某列设置为索引
df_demo.loc['Qiang Sun']  # 多人
df_demo.loc['Quan Zhao']  # 唯一
df_demo.loc['Qiang Sun', 'School']  # 返回Series
df_demo.loc['Quan Zhao', 'School']  # 返回标量
# 元素列表
df_demo.loc[['Qiang Sun', 'Quan Zhao'], ['School', 'Gender']]
# 切片操作；与上述Series的切片操作类似
df_demo.loc['Gaojuan You':'Gaoqiang Qian', 'School':'Gender']
df_loc_slice_demo = df_demo.copy()  # 拷贝
df_loc_slice_demo.index = range(df_demo.shape[0], 0, -1)  # 索引赋值
df_loc_slice_demo.loc[5:3]  # 取行索引数据
df_loc_slice_demo.loc[3:5]  # 没有返回值，说明不是整数位置的切片
# 布尔列表
df_demo.loc[df_demo.Weight > 70]  # 比较符号
df_demo.loc[df_demo.Grade.isin(['Freshman', 'Senior'])]  # isin()函数
# 对复合条件而言，可以用“|”（或）、“&”（且）和“～”（非）的组合来实现
condition_1_1 = df_demo.School == 'Fudan University'
condition_1_2 = df_demo.Grade == 'Senior'
condition_1_3 = df_demo.Weight > 70
condition_1 = condition_1_1 & condition_1_2 & condition_1_3
condition_2_1 = df_demo.School == 'Peking University'
condition_2_2 = df_demo.Grade == 'Senior'
condition_2_3 = df_demo.Weight > 80
condition_2 = condition_2_1 & (~condition_2_2) & condition_2_3
df_demo.loc[condition_1 | condition_2]
# df.select_dtypes(include, exclude) 选择哪些数据类型的列，排除哪些数据类型的列。

# %%
# 索引后赋值；一定是在一次索引后直接赋值；多次索引后赋值是赋值在copy上，会报SettingWithCopyWarning错
df_chain = pd.DataFrame([[0, 0], [1, 0], [-1, 0]], columns=list('AB'))
# df_chain[df_chain['A'] != 0]['B'] = 1 / df_chain[df_chain.A != 0].B = 1/ df_chain.loc[df_chain.A != 0]['B'] = 1
df_chain.loc[df_chain['A'] != 0, 'B'] = 1

# Series也可以使用loc，和dataframe用法一致。

# %%
import numpy as np

# iloc索引器；位置索引。
# *包含整数，整数列表，整数切片和布尔列表。
df_demo.iloc[1, 1]  # 第二行第二列
df_demo.iloc[[0, 1], [0, 1]]  # 前两行前两列
df_demo.iloc[1: 4, 2:4]  # 整数切片不包含结束端点
# 传入同长度的布尔序列
df_demo.iloc[np.isin(np.arange(df_demo.shape[0]), [1, 2, 3])]
# iloc使用布尔序列的时候，不能传入Series而必须传入序列的values；因此优先使用loc.
df_demo.iloc[(df_demo.Weight > 80).values].head()
# Series的iloc操作
df_demo.School.iloc[1]
df_demo.School.iloc[1:5:2]

# %%
# query()函数; 含有空格的的列名使用`列 名`表示
df.query('((School == "Fudan University")&'
         '(Grade == "Senior")&'
         '(Weight > 70))|'
         '((School == "Peking University")&'
         '(Grade!= "Senior")&'
         '(Weight > 80))')
# 可以使用and,or,is in, not in 替换&,|
# query中的 ==/!= 列表 表示的是is in not in的意思
df.query('Grade == ["Junior", "Senior"]')
# 加@符号，以引用外部变量。
query_list = ["Junior", "Senior"]
df.query('Grade == @query_list')

# %%
# 索引运算：交；并；差；并-交
df_set_1 = pd.DataFrame([[0, 1], [1, 2], [3, 4]], index=pd.Index(['a', 'b', 'a'], name='id1'))
df_set_2 = pd.DataFrame([[4, 5], [2, 6], [7, 1]], index=pd.Index(['b', 'b', 'c'], name='id2'))
id1, id2 = df_set_1.index.unique(), df_set_2.index.unique()
id1.intersection(id2)
id1.union(id2)
id1.difference(id2)
id1.symmetric_difference(id2)

# 将索引去掉
df_set_in_col_1 = df_set_1.reset_index()
df_set_in_col_2 = df_set_2.reset_index()
# 使用isin表示交集取数
df_set_in_col_1[df_set_in_col_1.id1.isin(df_set_in_col_2.id2)]
# 如上，将需要做集合操作的列设置为索引，然后使用集合操作得到Index Series，再通过loc选取数据。

# %%
# 多级索引 TODO

# %%
# 常用的索引方法；对数据进行操作
# 索引的交换（swaplevel(),reorder_levels());索引的删除(droplevel()) TODO
# 索引属性的修改(rename_axis(), rename()) TODO
# 索引的设置与重置(set_index(), reset_index())
df_new = pd.DataFrame({'A': list('aacd'), 'B': list('PQRT'), 'C': [1, 2, 3, 4]})
df_new.set_index('A')
df_new.set_index('A', append=True)  # 保留原来的索引
df_new.set_index(['A', 'B'])  # 指定多列索引
my_index = pd.Series(list('WXYZ'), name='D')
df_new = df_new.set_index(['A', my_index])  # 通过Series创建索引
df_new.reset_index(['D'])
df_new.reset_index(['D'], drop=True)  # 是否丢弃索引这列的数据
# 当所有索引都丢弃后，pandas会生成默认索引。

# %%
# 重构pandas索引（索引对齐）/ 常用在时间序列的数据中
# reindex；reindex_like
df_reindex = pd.DataFrame({"Weight": [60, 70, 80],
                           "Height": [176, 180, 179]},
                          index=['1001', '1003', '1002'])
df_reindex.reindex(index=['1001', '1002', '1003', '1004'], columns=['Weight', 'Gender'])

df_existed = pd.DataFrame(index=['1001', '1002', '1003', '1004'], columns=['Weight', 'Gender'])
df_reindex.reindex_like(df_existed)

# %%
# 4.分组
# df.groupby(分组依据)[数据来源].具体操作
df.groupby('Gender')['Weight'].mean()
df.groupby('Gender')['Weight'].median()  # 根据性别统计体重的中位数
# 多个分组变量，通过传入list
df.groupby(['School', 'Gender'])['Weight'].mean()
# 根据复杂逻辑分组；下面的布尔值分组
condition = df.Weight > df.Weight.mean()
df.groupby(condition)['Weight'].mean()
# 分组不一定按照已有数据，可以根据创建同样行个的数据进行分组
items = np.random.choice(list('abc'), df.shape[0])
df.groupby(items)['Weight'].mean()
# 条件组合
df.groupby([condition, items])['Weight'].mean()
# 通过drop_duplicates查看分组类别
df[['School', 'Gender']].drop_duplicates()
# df.groupby(['School', 'Gender'])['Weight'].mean()/df.groupby([df['School'], df['Gender']])['Weight'].mean()

# groupby对象；ngroups查看分组个数；groups索引到值的映射 （两个属性）
gb = df.groupby(['School', 'Gender'])
gb.ngroups
kv = gb.groups  # kv.keys() kv.values() kev.items()
# df.size返回长*宽个数； gb.size()返回每个分组的个数（两个方法）
# 通过get_group获取某组的数据
gb.get_group(('Fudan University', 'Female'))  # 返回pandas

# 分组的三大操作：聚合agg()；变换transform()；过滤filter()
# 聚合
# 内置：max()、min()、mean()、median()、count()、all()、any()、idxmax()、idxmin()、unique()、skew()、quantile()、sum()、std()和var()
gb = df.groupby('Gender')['Weight']
gb.idxmax()  # gb.idxmin() #返回最大/小值的是索引
gb.quantile(0.9)  # 返回9分位的值
# gb = df.groupby('Gender')[['Height', 'Weight']]
gb.max()  # 返回最大值
# agg()函数解决四个问题：无法同时使用多个函数；无法对特定的列使用特定的聚合函数；无法使用自定义的聚合函数；无法直接对结果的列名在聚合前进行自定义命名
# 使用多个函数
gb.agg(['sum', 'idxmax', 'skew'])
# 特定的列使用特定的函数
# gb.agg({'Weight':['sum', 'idxmax'], 'Height': 'count'})
# 使用自定义的函数;此处的x是Series
gb.agg(lambda x: x.max() - x.min())
gb.agg([lambda x: x.max() - x.min(), lambda x: x.max(), lambda x: x.min()])
gb.agg([lambda x: x.max() - x.min(), 'max', 'min'])


def my_func(s):
    res = 'High'
    # 某列分组均值与全局的均值比较
    if s.mean() <= df[s.name].mean():
        res = 'Low'
    return res


gb.agg(my_func)

# 聚合结果重命名;通过元组
gb.agg([('A', lambda x: x.max() - x.min()), ('B', lambda x: x.max()), ('C', lambda x: x.min())])
gb.agg([('A', lambda x: x.max() - x.min()), ('B', 'max'), ('C', 'min')])
# gb.agg({'Height': [('my_func', my_func), 'sum'], 'Weight': lambda x: x.max()})

# 变换
# 最常用的内置变换函数是累计函数：cumcount()、cumsum()、cumprod()、cummax()和cummin()
# 填充类变换函数：fillna()、ffill()和bfill()
example = pd.DataFrame({"A": list("aaabba"), "B": [3,6,5,2,1,7]})
example.groupby("A")["B"].cummax() #分组求累计的最大值
example.groupby("A")["A"].cumcount() #分组重新编号
# rank()用法 TODO

# transform
gb.transform(lambda x: (x - x.mean()) / x.std())
# 不支持多次变换？？？TODO
# gb.transform([lambda x: (x + x.mean())/ 2 * x.std(), lambda x: (x - x.mean()) / x.std()])
# 一次transform对指定的列使用不同的函数 TODO
# 组过滤 filter TODO;是按组进行迭代的

# agg和transform一样都是按列迭代的；filter和apply是按组进行迭代的；其实就是参数是series还是dataframe


