# 分割字符串
# 1.str.split()
# 2.re.split(); r'[]'; r'()' 返回list中含捕获的分割符，使用｜分割分割符; r'(?:)' 返回list中不含捕获的分割符，使用｜分割分割符
import re

line = 'asdf fjdk; afed, fjek,asdf, foo'

print(re.split(r'(;|,|\s)\s*', line))
# %%
# 字符串匹配
# str.startswith('匹配字符串') str.endswith('匹配字符串')
# str.startswith(('匹配字符串1', '匹配字符串2')) 必须是元组;list,set可以用tuple转
# 切片和正则都可以，但是匹配简短的字符串没有那么优雅
# str[-4:] == '匹配字符串'
# re.match('匹配字符串1｜匹配字符串2', str)
# 配合any或者all使用
# any(name.endswith(('匹配字符串1', '匹配字符串2')) for name in os.listdir('.'))

# %%
# shell通配符匹配字符串 （*.py;Dat[0-9]*.csv）
# fnmatch -> fnmatch() 根据操作系统确定是否对大小写敏感;fnmatchcase() 确定对大小写敏感
# fnmatch(str, 'shell通配符'); fnmatch('Dat45.csv', 'Dat[0-9]*')
from fnmatch import fnmatch

print(fnmatch('Dat45.csv', 'Dat[0-9]*.csv'))
# fnmatch()函数匹配能力介于简单的字符串方法和强大的正则表达式之间。
# 如果在数据处理操作中只需要简单的通配符就能完成的时候，这通常是一个比较合理的方案。
# 如果你的代码需要做文件名的匹配，最好使用glob模块。

# %%
# 字符串匹配和搜索

# str.startswith('匹配字符串');str.endswith('匹配字符串') 返回True or False
# str.find('匹配字符串') 返回索引位置

# re模块
import re
# True
re.match(r'\d+/\d+/\d+', '11/28a/2023') is None

# 同一模式做多次匹配
datepat = re.compile(r'\d+/\d+/\d+')
if datepat.match('11/28a/2023'):
    print('yes1')
elif datepat.match('11/28/2023'):
    print('yes2')
else:
    print('no')

# match(str);findall(str);finditer(str)
# match 总是从头开始匹配; 不管结尾
text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
datepat = re.compile(r'\d+/\d+/\d+')
m = datepat.match(text)
# True
print(m is None)
# False
print(re.match(r'\d+/\d+/\d+', '28/11/2023asdfsf') is None)
# 需要$截止符
print(re.match(r'\d+/\d+/\d+$', '28/11/2023asdfsf') is None)

# findall 找到字符串中有即可
par_list = datepat.findall(text)
# ['11/27/2012', '3/13/2013']
print(par_list)

# finditer 返回一个迭代器
par_iter = datepat.finditer(text)
# <class 'callable_iterator'> <re.Match object; span=(9, 19), match='11/27/2012'> <class 're.Match'>
print(type(par_iter), next(par_iter), type(next(par_iter)))

# 捕获
# 通过Match的group方法;m.group()匹配全部;m.group(1)捕获第一个括号;group(2)捕获第二个括号
# findall把捕获的内容存在元组里，返回list。
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')
m = datepat.match('28/11/2023')
print(m.group(), m.group(1), m.group(2), m.group(3))

pat_list = datepat.findall(text)
print(pat_list)

# 因为加了r''所以\不需要转义了。
# re级别的match和findall做一次简单的匹配可以，否则就需要先编译提高速度。

# %%
# 字符串搜索和替换
# str.replace('匹配字符串', '替换字符串')

# re模块的sub函数
# 1.re.sub('匹配模式', '替换模式', str)
# 2.先编译，然后匹配提高性能 c = re.compile('匹配模式')| c.sub('替换模式', str)
# 3.匹配模式中的捕获分组，可以用替换模式中\1 \2 \3匹配。
# 4.匹配模式通过?P<命名1>给分组命名；替换模式中通过\g<命名1>匹配
# 5.通过回调函数替代替换模式
# 6.subn返回替代的结果，还有替换的个数
text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
import re
print(re.sub(r'(\d+)/(\d+)/(\d+)', r'\3-\1-\2', text))
print(re.sub(r'(?P<month>\d+)/(?P<day>\d+)/(?P<year>\d+)', r'\g<year>-\g<month>-\g<day>', text))

from calendar import month_abbr
def change_date(m):
    mon_name = month_abbr[int(m.group(1))]
    return '{} {} {}'.format(m.group(2), mon_name, m.group(3))
print(datepat.sub(change_date, text))

newtext, n = datepat.subn(r'\3-\1-\2', text)
print(newtext, n, sep='\n')








