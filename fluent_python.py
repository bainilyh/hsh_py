import collections

# 特殊方法 __len__ __getitem__
# 特殊方法 TODO __setitem__(实现洗牌功能)

Card = collections.namedtuple('Card', 'rank suit')


class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                       for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):  # 提供[]访问元素；提供了切片操作；提供了可迭代的方式；提供in运算（实现__contains__方法)
        return self._cards[position]


deck = FrenchDeck()
print(len(deck))
# 使用random的choice来随机选择一张牌
from random import choice

print(choice(deck))
# 切片
print(deck[:3])
print(deck[-1])
# 迭代
for card in reversed(deck):
    print(card)
    break
# in
print(Card('Q', 'hearts') in deck)
print(Card('Q', 'bearts') in deck)
# 排序
suit_values = {'spades': 3, 'hearts': 2, 'diamonds': 1, 'clubs': 0}


def spades_high(card):
    # FrenchDeck.ranks 类字段
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]


for card in sorted(deck, key=spades_high, reverse=True):
    print(card)

# 如何使用特殊方法
# 特殊方法是被python解释器调用的，自己不用调用。
# for i in x -> iter(x) -> x.__iter__()
# 唯一亲自调用特殊方法的是__init__()

# doctest ddd理念 测试先行

# 加减乘除 __add__ __mul__
# 字符串表示 __repr__
# __abs__

from math import hypot


class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        # r是字符串的意思
        return 'Vector(%r, %r)' % (self.x, self.y)

    def __str__(self):
        # str;print 被使用
        pass

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):  # 标量乘向量
        return Vector(self.x * scalar + self.y + scalar)


# 容器序列（list, tuple, collections.deque)；存引用；可存多种类型
# 扁平序列(str, bytes, bytearray, memoryview, array.array)；存值；存一种类型

# 可变
# 不可变

# 新的序列类型，可从这两点先来判断

# list
# 列表推导（创建list）
symbols = '$¢£¥€¤'
codes = []
for symbol in symbols:
    codes.append(ord(symbol))

codes = [ord(symbol) for symbol in symbols]

# 与filter/map比较
[ord(symbol) for symbol in symbols if ord(symbol) > 127]
list(filter(lambda x: x > 127, map(ord, symbols)))

# 笛卡尔积
colors = ['black', 'white']
sizes = ['S', 'M', 'L']
[(color, size) for color in colors for size in sizes]

# 生成器表达式（创建序列）；迭代器，节约内存
tuple(ord(symbol) for symbol in symbols)
import array

array.array('I', (ord(symbol) for symbol in symbols))  # TODO 'I'是存储方式

# 生成器表达式计算笛卡尔积
for tshirt in ('%s %s' % (c, s) for c in colors for s in sizes):
    print(tshirt)

# 元组
# 当作记录
lax_coordinates = (33.9425, -118.408056)
city, year, pop, chg, area = ('Tokyo', 2003, 32450, 0.66, 8014)
traveler_ids = [('USA', '31195855'), ('BRA', 'CE342567'), ('ESP', 'XDA205856')]
for passport in sorted(traveler_ids):
    # 可以试一试'%r%r'
    print('%s/%s' % passport)

# 拆包（用于所有迭代序列上）；1.for循环可以拆包 _当作占位符；2.上例中的%也是拆包；3.125行的赋值也是拆包；4. *拆包
for country, _ in traveler_ids:
    print(country)

# 交换两个变量的值
a, b = 3, 4
a, b = b, a
print(a, b)

# 使用*把可迭代对象拆成函数的参数
t = (20, 8)
divmod(*t)

# 也是赋值拆包 split返回的是元组
import os

_, filename = os.path.split('/Users/bainilyhuang/temp/hello.py')

# *args用来捕获不确定数量的参数；这里用来捕获不确定拆包后的变量
a, b, *rest = range(5)
print(a, b, rest)
a, *rest, b = range(5)
print(a, rest, b)

# TODO 可以嵌套拆包

# 具名元组 collections.namedtuple
City = collections.namedtuple('City', ['name', 'country', 'population', 'coordinates'])
tokyo = City('Tokyo', 'JP', 26.999, (35.68, 139.69))
print(tokyo)
# 通过字段名字访问字段
print(tokyo.name, tokyo.coordinates)
# 类属性_fields 类方法_make(可迭代对象) 实例方法_asdict()
print(City._fields)

delhi_data = ('Delhi NCR', 'IN', 21.33, (28.61, 77.20))
delhi = City._make(delhi_data)  # 等于 City(*delhi_data)
print(delhi)

print(delhi._asdict())
for key, value in delhi._asdict().items():
    print(key, value, sep=':')

# 当作不可变列表 TODO 方法和属性列表


# 切片 list, tuple, str 等...
# 切片和区间会忽略最后一个元素

# s[a:b:c] s[::3] == s(slice(a,b,c)) s(slice(sep=3))

# 切片对象slice(a,b,c)

# TODO 多维切片和省略

# 给切片赋值
l = list(range(10))
print(l)
print(l[2:5])
# 赋值；少了数
l[2:5] = [20, 20]
print(l)
# 删除
del l[5:7]
print(l)
# 切片赋值对象必须是可迭代对象
l[2:5] = [100]
print(l)

# 序列的+和* 会创建新对象
# 创建列表的列表
board = [['_'] * 3 for i in range(3)]
print(board)
board[1][2] = 'X'
print(board)

# 下面复制了引用
weird_board = [['_'] * 3] * 3
print(weird_board)
weird_board[1][2] = 'X'
print(weird_board)

# TODO 序列的+=和*=


# list 排序
# list.sort() 本地排序 返回None表示数据进行了改变
# sorted(可迭代对象）返回一个新的可迭代序列
# 参数 reverse, key

# bisect 管理已经排序的序列
# 在已经排序的序列中 bisect.bisect查找和bisect.insort插入元素；二分查找
from bisect import bisect, insort, bisect_left
import sys

# bisect (可以在很长的序列中，作为index的替代）
HAYSTACK = [1, 4, 5, 6, 8, 12, 15, 20, 21, 23, 23, 26, 29, 30]
NEEDLES = [0, 1, 2, 5, 8, 10, 22, 23, 29, 30, 31]
# bisect是bisect_right别名；lo/hi限制范围；
bisect(HAYSTACK, 31)  # 返回序列中31应该的索引
# bisect_left返回相同元素的左边索引；而bisect/bisect_right返回相同元素的右边索引
bisect_left(HAYSTACK, 31)


# bisect 建立以数字作为索引的查询表格
#  60 ,70, 80, 90  ->  FDCBA
def grade(score, breakpoints=[60, 70, 80, 90], grades='FDBCA'):
    i = bisect(breakpoints, score)
    return grades[i]


[grade(score) for score in [33, 99, 77, 70, 89, 90, 100]]

# insort
# 同样提供lo/hi两个参数
import random

my_list = []
for i in range(14):
    new_item = random.randrange(28)
    insort(my_list, new_item)
    print(my_list)

# bisect可以用在包括list, tuple, str等所有序列上！
# 数组；如果只是数字用数组更高效 array.array；
# 支持所有序列操作 pop, insert, extend
# 支持读写 frombytes tofile
# b 有符号的字符 -128 到 127；d 浮点数

# 生成浮点数数据，存入文件
from array import array
from random import random

floats = array('d', (random() for i in range(10 ** 7)))
floats[-1]
# 写入文件
with open('./floats.bin', 'wb') as fp:
    floats.tofile(fp)

# 读取
floats = array('d')
with open('./floats.bin', 'rb') as fp:
    floats.fromfile(fp, 100)
floats[0]

# 一般都是用pickle
# list 和 数组的api对比
# python3.4去掉 数组.sort()方法；sorted(floats) 返回list
floats = array(floats.typecode, sorted(floats))
floats

# 内存视图 memoryview 不复制内容，但可以操作一个数据的不同切片
# memoryview.cast
# h 有符号的短整型，每个数字是2个字节；B是无符号的字符，每个字符一个字节。
numbers = array('h', [-2, -1, 0, 1, 2])
memv = memoryview(numbers)
len(memv)
memv_oct = memv.cast('B')
# 要显示memv_out中的内存，使用tolist方法
print(len(memv_oct), memv_oct.tolist(), sep='\n')
# 修改numbers中第5个字节的值
memv_oct[5] = 1
print(memv_oct.tolist())
# 显示numbers中的变化
print(numbers)

# TODO 简介numpy和scipy

# 双向队列和其他队列
# 列表的append pop 可以当作栈和队列使用
# 列表的append pop(0) 可以当作先进先出的队列；但是列表移动第一个，非常耗时。

# collections.deque是双向队列高效的实现
from collections import deque

# dp = deque(maxlen=10)
dq = deque(range(10), maxlen=10)
print(dq)
dq.rotate(4)  # 右边4个旋转到前面；原地修改
print(dq)
dq.rotate(-5)  # 左边5个旋转到后面
print(dq)
# append;appendleft;extend(可迭代对象);extendleft(可迭代对象);pop; popleft

# deque对删除数据会慢一些，因为它只对头尾操作做了优化。
# deque方法是原子操作，可以用于多线程（当作栈使用，不用担心资源锁）。
# TODO 和列表的api对比

# Queue;LifoQueue;PriorityQueue
# 参数是maxsize

# 包multiprocessing实现了Queue；用于进程通信。
# multiprocessing.JoinableQueue 用于任务管理。

# 包asyncio提供Queue;LifoQueue;PriorityQueue;JoinableQueue;用于异步编程

# TODO heapq


# 【字典；集合】
from collections.abc import Mapping
from collections.abc import MutableMapping

# 非抽象类一般不会继承这两个抽象类，而是对dict或者collections.User.Dict进行扩展
my_dict = {}
print(isinstance(my_dict, Mapping))
# 可散列对象必须实现__hash__和__qe__方法；且散列值不变。

# 可散列类型：str, bytes, 数值类型，frozenset，元组（只包含可散列对象）

# 构建字典的不同方式:
a = dict(one=1, two=2, three=3)
b = {'one': 1, 'two': 2, 'three': 3}
c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
d = dict({'one': 1, 'two': 2, 'three': 3})
print(a == b == c == d)

# 字典推导：任何以健值对作为元素的可迭代对象中构造字典。
DIAL_CODES = [(86, 'China'), (91, 'India'), (1, 'United States'), (62, 'Indonesia'), (55, 'Brazil'), (92, 'Pakistan'),
              (880, 'Bangladesh'), (234, 'Nigeria'), (7, 'Russia'), (81, 'Japan'), ]
country_code = {country: code for code, country in DIAL_CODES}
print(country_code)
{code: country.upper() for country, code in country_code.items() if code < 66}

# 常见的映射方法
# dict defaultdict OrderedDict api比较 TODO
d = dict()
d['key']  # 报错
d.get('key', 0)  # 没有给予默认值

# 更新某key 1 （不好的实现）
occcurences = d.get('a', [])
occcurences.append(1)
d['a'] = occcurences
print(d)

# 更新某key 2
d.setdefault('b', []).append(10)
print(d)

# 更新某key 3 （相当于2，但是查询key变多很多）
if 'b' not in d:
    d['b'] = []
d['b'].append(1)
print(d)

# 映射不存在的两种实现方式
# 1.collections.defaultdict 2.定义dict子类，实现__missing__方法
from collections import defaultdict

dd = defaultdict(list)
dd['a'].append(1)
print(dd)
# 说明：dd['a']解释器是调用__getitem__但是dd.get('a')不会
# defaultdict中__getitem__会处理没有key的情况
print(dd.get('c'))  # None
print(dd['c'])  # []

# dict变种
# collections.OrderedDict key是加入时候的顺序
# .popitem()返回最后一个元素
# .popitem(last=False)返回第一个元素
from collections import OrderedDict

d = OrderedDict({'a': 1, 'b': 2, 'c': 3})
print(d)
print(type(d.popitem()))
print(d)

# collections.ChainMap;存放不同Map
from collections import ChainMap
import builtins

pylookup = ChainMap(locals(), globals(), vars(builtins))
print(len(pylookup))

# collections.Counter 整数计数器; + - 合并记录；most_common([n])返回最常见的n个健和他的记录
from collections import Counter

ct = Counter('abracadabra')
print(ct)
ct.update('aaaaazzz')
print(ct)
print(ct.most_common(2))

# TODO
# collections.UserDict;使用python重新实现了dict;需要用户子类化;data属性是dict对象
# 一般实现自定义的字典需要继承UserDict而不是dict。

# 不可变映射类型
# types.MappingProxyType只读代理映射
# MappingProxyType是动态的，对d的改动会映射到MappingProxyType上
from types import MappingProxyType

d = {1: 'A'}
d_proxy = MappingProxyType(d)
print(d_proxy[1])
# d_proxy[2] = 'B'


# 集合 Set
a = {1, 2}
a = set([1, 2])  # 比上面慢
# 空集 不能是 a={} 被字典暂用了
a = set()
# 输出元素
print(a.pop())

# frozenset 没有字面量；先构造list再转换集合；不可变集合
a = frozenset(range(10))
print(a)
b = frozenset(a)
print(b)

# 集合推导
# 获取字符的名字
from unicodedata import name

{chr(i) for i in range(32, 256) if 'SIGN' in name(chr(i), '')}

# 集合api TODO
from collections import Set
from collections import MutableSet

a = {'a', 'b'}
b = {'c', 'd'}
a.union(b)
a.union('e')

# 集合数学运算 TODO

# set dict list效率
## 使用dict.fromkeys(字典)构造 dict
# 字典中的散列
# dict实现..
## 不能边遍历边修改；字典耗内存很大，所以最好以元组或者具名元组存储大数据。
## keys(), items(), values() 返回字典视图。
# set实现..
## 集合也耗内存

# 字典中的update可以批量更新键值对


# 【文本；字节序列】
# 字符不是str中的字符，而是unicode中的字符。
# 字符的标识，也就是码位。
# 字符的具体表述取决于所使用的编码。（utf-8, utf-16le)
# 编码：码位 -> 字节序列
# 解码：字节序列 -> 码位

s = 'café'
print(len(s))  # 四个码位
# bytes类型 字面量以b开头
b = s.encode('utf-8')  # 编码
print(len(b), b, type(b))
c = b.decode('utf-8')  # 解码
print(len(c), c, type(c))

# TODO 字节概要；基本的编解码器；了解编解码问题

# 处理文本文件 bytes -> str(处理字符串) -> bytes
with open('./test.txt', mode='w', encoding='utf-8') as fp:
    fp.write('café')

with open('./test.txt', mode='r') as fp:
    # TextIOWrapper
    print(type(fp), fp)
    print(fp.read())

# 查看一个文件有多少个字节
# 可见é占了两个字节
import os

print(os.stat('./test.txt').st_size)

# mode中 b 是二进制的意思
with open('./test.txt', mode='rb') as fp:
    # BufferedReader
    print(type(fp), fp)
    b = fp.read()
    print(len(b), b)


# TODO 编码默认值：一团糟
# TODO 为了正确比较而规范化Unicode字符串
# TODO ...


# 【一等函数】：运行中创建；能赋值给遍历，或是数据结构中的元素；作为函数的参数；作为函数的返回值。
# 函数是类(function)的实例
def factorial(n):
    """return n!"""
    return 1 if n < 2 else n * factorial(n - 1)


print(type(factorial))
print(factorial.__doc__, help(factorial), sep='\n')

# 赋值
fact = factorial
# 调用
print(fact(5))
# 作为参数传递
print(list(map(fact, range(11))))
# 高阶函数；map；filter；reduce；sorted(key,..) key是接受单参数的函数
print(list(map(fact, filter(lambda x: x % 2, range(6)))))
# 列表推导替代了高阶函数；reduce模块移到了functools模块中了；求和用sum替换了reduce
from functools import reduce
from operator import add

print(reduce(add, range(100)))

print(sum(range(100)))
# all（可迭代对象）；any（可迭代对象）

# 匿名函数 lambda关键字
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
print(fruits)
print(sorted(fruits, key=lambda word: word[::-1]))


# TODO 可调用对象
# TODO 用户定义的可调用类型
# TODO 函数内省

# 定位参数；仅限关键字参数；*和**
# *content是tuple； attrs是dict
def tag(name, *content, cls=None, **attrs):
    """生成一个或者多个html标签"""
    if cls is not None:
        attrs['class'] = cls

    if attrs:
        attr_str = ''.join(' %s="%s"' % (attr, value)
                           for attr, value
                           in sorted(attrs.items()))
    else:
        attr_str = ''

    if content:
        return '\n'.join('<%s%s>%s</%s>' % (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s />' % (name, attr_str)


print(tag('br'))
print(tag('p', 'hello'))
print(tag('p', 'hello', 'world'))
print(tag('p', 'hello', id=3))
print(tag('p', 'hello', 'world', cls='sidebar'))
# 位置参数，可以用关键词传参；这里的content并没有被*content捕获，而是被**attrs捕获
print(tag(content='testing', name='img'))
# 通过拆包传参数;这里通过**拆包
import collections
attr = collections.OrderedDict({'id': 33, 'num': 55})
print(tag(name='image', **attr))



