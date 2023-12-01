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

# 字符串忽略大小写的搜索替换
# re.IGNORECASE
# 字符大小写转化 str.upper() str.lower()
text = 'UPPER PYTHON, lower python, Mixed Python'
print(text)
print(re.findall('python', text, flags=re.IGNORECASE))
print(re.sub('python', 'snake', text, flags=re.IGNORECASE))
# TODO 需要辅助函数来处理Python -> Snake问题

# %%
# 最短匹配模式
# 增加匹配的内容上增加？
# 通过在 * 或者 + 这样的操作符后面添加一个 ? 可以强制匹配算法改成寻找最短的可能匹配。
import re

text = 'Computer says "no." Phone says "yes."'
str_pat = re.compile('"(.*?)"')  # ()捕获内容
print(str_pat.findall(text))

# 多行匹配模式
# .不能匹配换行符
# 1.使用re.DOTALL
# 2. 使用?:不捕获(); ((?:.|\n)*?)
str_pat = re.compile(r'/\*(.*?)\*/', flags=re.DOTALL)
text = '''/* this is a 
 multiline comment */
'''
print(str_pat.findall(text))

str_pat2 = re.compile(r'/\*((?:.|\n)*?)\*/')
print(str_pat2.findall(text))

# %%
# 删除字符串中不需要的字符
# 1.str.strip(); str.lstrip(); str.rstrip()
# 2.可以指定去掉的字符 str.strip('去掉的字符')
# 3.删除中间等的字符使用replace str.replace('去掉的字符', '替换的字符')
# 4.删除中间等的字符使用re.sub
# 5.与迭代器配合读取文件，非常高效
with open('./torch/deep_learning_pytorch0.py') as f:
    lines = (line.strip() for line in f)
    for line in lines:
        print(line)
        break
# %%
# 字符串对齐
# 1.str.ljust() str.rjutst() str.center()
text = 'Hello World'
print(text.rjust(20, '='))
# 2.format > < ^
# >20 *>20
print(format(text, '>20'))
print(format(text, '*^20'))
# 3.多值format
print('{:>10s} {:>10s}'.format('hello', 'world'))
# 4.format格式化数值
x = 1.2345
print(format(x, '>10'))
print(format(x, '>10.3f'))
# TODO 5.more format

# %%
# 1.序列或者迭代器中合并字符串 join
parts = ['Is', 'Chicago', 'Not', 'Chicago?']
print(','.join(parts))
# 2.少数合并使用+ 或者 format也行
# 3.思考如下代码的使用场景
# ## Version 1 (string concatenation)
# f.write(chunk1 + chunk2)
#
# # Version 2 (separate I/O operations)
# f.write(chunk1)
# f.write(chunk2)
# %%
# 字符串中插入变量
# format
s = '{name} has {n} messages.'
print(s.format(name='Guido', n=37))
# format_map() var()
name = 'Guido'
n = 37
print(s.format_map(vars()))
# 只要实力对象中有的字段，也可以
class Info:
    def __init__(self, name, n):
        self.name = name
        self.n = n

a = Info('Guido', 37)
print(s.format_map(vars(a)))

# %%
# 数字的四舍五入