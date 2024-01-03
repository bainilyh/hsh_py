# 以下代码是asyncio的核心概念
import asyncio
import time


async def say_after(delay, what):
    await asyncio.sleep(delay)
    # print(what)
    return f'{what} - {str(delay)}'


async def main():
    start = time.perf_counter()
    print(f'started at {start}')
    await say_after(1, 'hello')
    await say_after(2, 'world')
    end = time.perf_counter()
    print(f'ended at {end}, cost:{end - start}')


# asyncio.run(main())


# started at 3.595986986
# hello
# world
# ended at 6.601625416, cost:3.00563843

# 创建一个even loop空间，显示的运行任务，所以不存在竞争的关系。
# async def 函数名称；创建一个协程函数；调用协程函数返回协程对象。
# 通过asyncio.run(协程对像)将协程对象转换为任务，并开始执行这个任务。
# even loop机制不会主动控制任务，而是由任务主动交控制权；由两种方式交回控制权：任务显示await一个协程对象；任务完成。
# 这里await 协程对象还有一个功能，告诉even loop，又给你添加了一个任务；下面实现不用await添加任务，可以做到真正的协程。

async def main2():
    task1 = asyncio.create_task(say_after(1, 'hello'))  # 给even loop添加个task，控制权还在main
    task2 = asyncio.create_task(say_after(2, 'world'))  # 给even loop添加个task，控制权还在main
    start = time.perf_counter()
    print(f'started at {start}')
    # main让出控制权，此时even loop有三个任务，task1拿到控制权后，通过await添加了一个sleep(1)任务并让出控制权
    # 此时even loop任务有四个任务。
    # task2拿到控制权全后，通过await添加了一个sleep(2)任务并让出控制权。
    # 此实even loop任务有五个任务。
    # 当sleep(1)任务执行完成后，控制权还给了task1，打印后结束任务交出控制权给main，main交出控制权给task2。
    # 当sleep(2)任务执行完成后，控制权还给了task2，打印后结束任务交出控制权给main，main完成任务。even loop为空。
    ret1 = await task1
    print(ret1)
    ret2 = await task2
    print(ret2)
    end = time.perf_counter()
    print(f'ended at {end}, cost:{end - start}')


# started at 2.995650375
# hello
# world
# ended at 5.00080444, cost:2.0051540649999997
# asyncio.run(main2())

async def main3():
    # task1 = asyncio.create_task(say_after(1, 'hello'))
    # task2 = asyncio.create_task(say_after(2, 'world'))
    start = time.perf_counter()
    print(f'started at {start}')

    # gather可以传递task,也可以传递协程对象，还可以传递Feature就是gather的返回对象。
    # ret = await asyncio.gather(task1, task2)
    ret = await asyncio.gather(say_after(1, 'hello'), say_after(2, 'world'))
    print(ret)  # 返回list
    end = time.perf_counter()
    print(f'ended at {end}, cost:{end - start}')


asyncio.run(main3())
