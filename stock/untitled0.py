#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:15:47 2023

@author: bainilyhuang
"""

def factorial(n):
    """returns n!"""
    return 1 if n < 2 else n * factorial(n - 1)

# 函数的属性 dir(函数名)
# 1. __doc__ 可以直接访问；也可以通过help访问

# 高阶函数(函数shi)
# map(单参数函数, 可迭代对象), sorted(可迭代对象, key=单参数函数)
# filter, reduce, --apply(废弃，用于不定量的参数, 替换方法fn(*args, **keywords))
# # # #
# 列表推导和生成器表达式多数可以替换map,filter
# # # # 
# reduce 在 functools包中 operator包中有操作符; 一般用sum效率高
from functools import reduce
from operator import add
print('reduce function result:', reduce(add, range(10)), sep='\t')
print('sum function result:', sum(range(10)), sep='\t')
# all(iterable) 和 any(iterable)
#%%
import random
class BingoCage:
    
    def __init__(self, items):
        self._items = list(items)
        random.shuffle(self._items)
        
    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage')
    
    def __call__(self):
        return self.pick()
#%%
class C:
    pass
c = C()
def func():
    pass
a = set(dir(func)) - set(dir(c))
#%%
# 装饰器
def f(fuc):
    print('my_f')
    return max

@f
def g():
    return 'a'
#%%
b = 10
def f1(a):
    global b
    print(a)
    print(b)
    b = 4
f1(3)

