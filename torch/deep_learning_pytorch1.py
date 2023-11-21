# 卷积
# M百万 B十亿
# 一张照片12M像素；RGB图片36M元素；一层100个大小的隐藏层就有3.6B的数据。

# 图片特性的两个原则：平移不变形；局部性。

# 将输入和输出变形为矩阵
# 原本输入是一维的；现在是三维度（通道；宽度；高度）

# TODO 不懂

# 对全连接层使用平移不变形和局部性就是卷积层。

# 二维交叉相关；输入（矩阵） * 卷积核（矩阵） = 平移的对应元素相乘相加
# 二维卷积层；输入X(nh,nw)；核W(kh,kw)；偏差 b
# 输出Y(nk -kh + 1, nw - kw + 1) = X交叉相关W + b
# W和b是参数
# 不同的卷积核提取不同的特征

# 交叉相关和卷积仅仅只是W参数位置不同（左右、上下翻），由于是要学的参数，所以实际应用中没有区别。

# 一维 w是向量；文本；语言；时序序列
# y(i) = sigma(a=1->h)(w(a) * x(i + a))

# 二维；图片

# 三维；视频；医学图像；气象地图
# y(i, j ,k) = sigma(a=1->h)sigma(b=1->w)sigma(c=1->d)(w(a,b,c) * x(i+a, j+b, k+c))

# 总结：卷积层是将输入和核矩阵交叉相关运算，加上偏移得到输出；核矩阵和偏移可学习；核矩阵的大小是超参数。

# 图片卷积
# 互相关运算
import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape  # 行数，列数
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


# 卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias


# 检测边缘demo
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)

# 检测横向边缘
print(X.t())  # tensor.t()转至

# 学习卷积核
# 1输入通道 1输出通道
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
# （批量⼤⼩、通道、⾼度、宽度）
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2
for i in range(20):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
print(conv2d.weight.data, conv2d.weight.data.shape, conv2d.weight.data.reshape(1, 2))

# 课后问答
# 卷积有完整的数学定义；深度神经网络借用过来了。


# 卷积层：填充和
# 填充：使用卷积网络越深，输出越小。
# (n(h) - k(h) + p(h) + 1) * (n(w) - k(w) +p(w) + 1)
# 通常p(h) = k(h) - 1; p(w) = k(w) - 1
# 当k(h)为奇数：上下填充p(h)/2

# 步幅：输出是和卷积核大小线性相关；输入越大，需要越深的网络才能得到越小的输出。
# L((n(h) - k(h) + p(h) + s(h))/s(h)) * L((n(w) - k(w) + p(w) + s(w))/s(w))

# 填充和步幅、卷积核都是超参数


import torch
from torch import nn


def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


# padding是一边填充大小, 这里相当于p(h)=p(w)=2
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(8, 8)
Y = comp_conv2d(conv2d, X)
print(Y.shape)

# p(h) = 4, p(w) = 2
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

# L((8 + 1) / 2) = 4
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
Y = comp_conv2d(conv2d, X)
print(Y.shape)

# (8 - 3 + 0 + 3) / 3, (8 - 5 + 2 + 4) / 4; 不对称一般不用
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
Y = comp_conv2d(conv2d, X)
print(Y.shape)

# 课后问题
# 填充一般是k(核)-1;通常步幅是1或者2；卷积核最重要，一般3*3
# 一般不自己设计网络结构；网络结构相对影响很小；数据预处理比较关键；
# NAS（有钱人的游戏）；让超参数一起训练；autogulon？


# 输入和输出的通道
# 多个输入通道 RGB转灰色会丢失信息
# 每个通道都有自己的卷积核；算出来的输出按元素相加。
# X: c(i) * n(h) * n(w)
# K: c(i) * k(h) * k(w)
# Y: m(h) * m(w)

# 多个输出通道
# X: c(i) * n(h) * n(w)
# K: c(o) *c(i) * k(h) * k(w)
# Y: c(o) * m(h) * m(w)


# 每个输出通道核可以识别特定的模式
# 输入通道核识别并组合输入中的模式


# 1*1卷积层：融合不同通道的信息。
# 相当于输入n(h)n(w) * c(i) ，权重c(o) * c(i)的全连接层。


# 二维卷积层
# X: c(i) * n(h) * n(w)
# K: c(o) * c(i) * k(h) * k(w)
# B: c(o) * c(i)
# Y: c(o) * m(h) * m(w)

# 计算复杂度O(c(i)c(o)k(h)k(w)m(h)m(w))
# ci = co = 100
# k(h) = k(w) = 5
# m(h) = m(w) = 64
# 10亿次运算 = 1G FLOP
# 10层；100万样本，10P次FLOP（浮点计算）
# cpu 18h gpu 14min （单纯扫一遍数据（不算梯度计算））

# 代码实现
import torch
from d2l import torch as d2l


# 计算多通道的输入
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor(
    [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

# 计算多通道的输出
K = torch.stack((K, K + 1, K + 2), 0)  # 在0维度累加起来
print(K.shape)


def corr2d_multi_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


print(corr2d_multi_out(X, K))


# 验证1*1卷积等价于全链接
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape(c_i, h * w)
    K = K.reshape(c_o, c_i)
    Y = torch.matmul(K, X)
    return Y.reshape(c_o, h, w)


X = torch.normal(0, 1, size=(3, 3, 3))
K = torch.normal(0, 1, size=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_out(X, K)

print(torch.abs(Y1 - Y2).sum() < 1e-6)

# torch卷积的调用
from torch import nn

# 输入通道 输出通道
conv2d = nn.Conv2d(3, 2, kernel_size=1, padding=0, stride=1)
nn.init.zeros_(conv2d.bias.data)
print(conv2d(X))
K1 = conv2d.weight.data
print(corr2d_multi_in_out_1x1(X, K1))

# 课后问答
# 输入减半的时候，一般增加一倍的通道数
# feature_map指卷积的输出


# 池化层
# 卷积对位置太敏感；加池保持【平移不变性】
# 二维最大池化层；平均池化层
# 超参数：窗口大小；填充；步幅；通道数和前一个输入的通道数一样。

# 代码实现
import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), mode='avg'))

# torch实现
X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
# 步幅和池化窗口大小相同，所以这里特殊设置stride=1
pool2d = nn.MaxPool2d(3, padding=0, stride=1)  # 3是3*3的大小
print(pool2d(X))

# 设置2个通道
X = torch.cat((X, X + 1), dim=1)
print(pool2d((X)))

# 课后问答
# 池化层在卷积的后面，是的卷积提取的特征对位置不那么敏感。
# 现在池化层用的越来越少；数据增强了；stride可以放入卷积层（池化的另一个作用是降低输出维度）
# 输入 -> 卷积层 -> 池化层 -> 卷积层 -> 池化层 -> 拉平 -> 全连接 -> 全连接 -> softmax(高斯)

# LeNet (80年代末提出）
# MNIST数据集；5万个训练数据；1万个测试数据；图像28 *28；类别是10

# 代码实现
import torch
from torch import nn
from d2l import torch as d2l


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)  # view 和 reshape差不多


net = nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),  # 28 - 5 + 1 + 4 = 28
    nn.AvgPool2d(2, stride=2),  # 28 / 2 = 14
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 14 - 5 + 1 = 10
    nn.AvgPool2d(2, stride=2),  # 10 /2 = 5
    nn.Flatten(),  # 16 * 5 * 5
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:  # net是由Sequential实现所以可以迭代
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

# 训练
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# # 修改evaluate_accuracy
# def evaluate_accuracy_gpu(net, data_iter, device=None):
#     """计算整个数据集上的精度;使用GPU"""
#     if isinstance(net, nn.Module):
#         net.eval()  # 进入评估模式，不计算梯度
#         if not device:
#             device = next(iter(data_iter)).device
#     metric = d2l.Accumulator(2)
#     with torch.no_grad():
#         for X, y in data_iter:
#             if isinstance(X, list):
#                 X = [x.to(device) for x in X]
#             else:
#                 X = X.to(device)
#             y = y.to(device)
#             metric.add(d2l.accuracy(net(X), y), y.numel())  # y.numel()y的个数
#     return metric[0] / metric[1]

lr, num_epochs = 0.5, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# V100
# loss 0.477, train acc 0.819, test acc 0.802
# 79484.8 examples/sec on cuda:0

# mlp
lr, num_epochs = 0.03, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Linear(num_hiddens2, num_outputs)
)
# v100
# loss 0.402, train acc 0.858, test acc 0.822
# 105906.8 examples/sec on cuda:0


# 课后问答
# cnn explaniner 查看卷积学习的内容
# 权重visualizaition 好看但没用


# AlexNet （2012深度学习）
# 2000年最火的是核方法：特征提取；选择核函数计算相关性；凸优化问题；漂亮的定理。
# 计算机视觉：特征工程（SIFT SURF） -> 简单的分类模型 SVM等

# 数据集；ImageNet（2010）
# cs231n
# AlexNet:更大更深的LeNet；全连接层加丢弃法，Relu，MaxPooling；数据增强；改变了机器学习的方式
# 架构：
# -> 3*224*224 -> 11*11 conv 96通道 stride 4 -> 3 * 3 MaxPool stride 2
# -> 5*5 conv pad 2(pad=2输入=输出 因为：n - 5 + 1 + 2*pad） 256个通道 -> 3*3 MaxPool stride 2
# -> 3 * 3 conv pad 1 384个通道
# -> 3 * 3 conv pad 1 384个通道
# -> 3 * 3 conv pad 1 384个通道
# -> 3*3 MaxPool stride 2
# -> Dense(4096)
# -> Dense(4096)
# -> Dense(1000) 输出1000个分类

# 复杂度对比：AlexNet参数复杂度.png
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),  # 54
    nn.MaxPool2d(kernel_size=3, stride=2),  # 26 (54 - 3 + 2)/ 2
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),  # 26
    nn.MaxPool2d(kernel_size=3, stride=2),  # 12
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),  # 12
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),  # 12
    # nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), # 12
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),  # 12
    nn.MaxPool2d(kernel_size=3, stride=2),  # 5
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape: \t', X.shape)

batch_size = 128
# resize 只是模拟；真实情况下不能这么干
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# GPU Tesla V100-SX
# loss 0.329, train acc 0.878, test acc 0.878
# 3108.8 examples/sec on cuda:0


# VGG（2013）：更深更大
# 选项：
# 更多的全连接层？
# 更多的卷积层？
# 将卷积层组合成块 VGG块 3*3卷积
# 将AlexNet的卷积层替换成n个VGG块串联；串联思想形成。
# https://cv.gluon.ai/model_zoo/classification.html

# vgg11
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        ))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 设置5块vgg
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),  # 每个块输出/2，输入224
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )


net = vgg(conv_arch)
X = torch.rand(size=(1, 1, 224, 224))
# 数据减半->通道数加倍
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape: \t', X.shape)

# 由于vgg11计算量更大，构建一个通道数比较少的网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# conv_arch  = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# GPU Tesla V100-SX
# loss 0.163, train acc 0.939, test acc 0.922
# 494.0 examples/sec on cuda:0


# NiN（网络中的网络）
# 卷积的参数：c(i) * c(o) * k**2
# 卷积后第一个全连接的参数:
# LeNet 16 * 5 * 5 * 120 = 48K
# AlexNet 256 * 5 * 5 * 4096 = 26M
# VGG 512 * 7 * 7 * 4096 = 102M
# NiN不用全连接层;用1*1卷积层相当于1个全连接层; NiN块：一个卷积层后加两个1*1卷积
# 总结：
# 1.NiN块使用卷积层+2个1*1卷积层；后者对每个像素增加非线性
# 2.NiN使用全局平均池化层来替代VGG和AlexNet中的全连接层
# 3.AlexNet中的全连接层；不容易果泥和，更少的参数个数


import torch
from torch import nn
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


# 是对AlexNet改良...
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),  # 54
    nn.MaxPool2d(3, stride=2),  # 26
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),  # 26
    nn.MaxPool2d(3, stride=2),  # 12
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),  # 12
    nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),  # 5
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),  # 5
    nn.AdaptiveAvgPool2d((1, 1)),  # (1,1)->高宽都变成1
    nn.Flatten()
)

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# GPU Tesla V100-SX
# loss 0.341, train acc 0.872, test acc 0.882
# 2520.7 examples/sec on cuda:0


# 课后问答
# pytorch上线 -> torch script -> c++ 不建议
# 为什么没有softmax层，因为他用在了training中？train_ch6 -> CrossEntropyLoss
# 修改预测代码 d2l.predict_ch3()


# GoogLeNet（含并⾏连结的⽹络）（卷积层个数>100)
# Inception块：解决选择1*1卷积还是3*3，5*5卷积的困局
# 4个路径从各个层面抽取信息，然后在输出通道上合并。
# Inception降低了参数，也在增加了通道的多样性

# Inception变种
# InceptionV2 - 使用BN
# InceptionV3 - 在V2基础上修改Inception快
#     *替换5*5为多个3*3卷积层
#     *替换5*5为7*1和1*7卷积层
#     *替换5*5为3*1和1*3卷积层
#     *更深
# InceptionV4 - 使用残差连接

# 总结：
# 1.使用4条有不同超参数的卷积层和池化层来抽取不同的信息;模型小，计算复杂度低
# 2.GoogleNet使用9个Inception块，第一个达到上百层网络（非深度）；后续一系列改进优化了模型

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最⼤汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# loss 0.243, train acc 0.908, test acc 0.892
# 3296.9 examples/sec on cuda:0


# 批量归一化（卷积神经网络）
# 损失出现在最后，越后的层训练越快；
# 离数据越近层权重训练越慢；离数据越近的权重一动，后面权重还是得调整；导致多次学习；收敛慢；

# 能不能在改变底部（离数据越近的层）权重的时候避免再去更新顶部权重？

# 数据的方差和均值在每一层之间会发生变化

# 批量归一化：加入收敛速度；但是不改变模型精度
# 阿尔法、贝塔是参数，需要学习；u(B)表示在B这个批量上的均值；o(B)是表示在B这个批量的方差的开根号；
# x(i + 1) = 阿尔法 * (x(i) - u(B)) / o(B) + 贝塔

# 参数：阿尔法、贝塔
# 作用在全连接或者卷积层后，激活函数前；或者全连接层和卷积层输入上？
# 全连接层，作用在特征维
# 卷积层，作用在通道维？

# 每个小批量离加入噪音来控制模型复杂度
# x(i + 1) = 阿尔法 * (x(i) - u(B)) / o(B) + 贝塔
# u(B)随机偏移；o(B)随机缩放

# 没必要和dropout混合使用。

# 从零实现
# moving_mean推理时候整个数据集的均值；moving_var同理；
# eps;moment固定值；不要调试
import torch
from torch import nn
from d2l import torch as d2l

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, moment):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 2表示全连接；4表示卷积
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0) # 特征维度
            var = ((X - mean)**2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = moment * moving_mean + (1.0 - moment) * mean
        moving_var = moment * moving_var + (1.0 - moment) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, moment=0.9
        )
        return Y

# 应用在LeNet上
net = nn.Sequential(
    # Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), BatchNorm(6, num_dims=4), nn.Sigmoid(),  # 28 - 5 + 1 + 4 = 28
    nn.AvgPool2d(2, stride=2),  # 28 / 2 = 14
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),  # 14 - 5 + 1 = 10
    nn.AvgPool2d(2, stride=2),  # 10 /2 = 5
    nn.Flatten(),  # 16 * 5 * 5
    nn.Linear(16 * 5 * 5, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
lr, num_epochs = 1.0, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# loss 0.245, train acc 0.909, test acc 0.864
# 38191.1 examples/sec on cuda:0

# 随便看一眼gamma和beta
print(net[1].gamma, net[1].beta, sep='\n')


# 简洁实现
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6), nn.Sigmoid(),  # 28 - 5 + 1 + 4 = 28
    nn.AvgPool2d(2, stride=2),  # 28 / 2 = 14
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),  # 14 - 5 + 1 = 10
    nn.AvgPool2d(2, stride=2),  # 10 /2 = 5
    nn.Flatten(),  # 16 * 5 * 5
    nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))


# 课后问答
# xavier是在权重初始化的时候做归一化；但是BN是在训练的每一层。
# BN使用在深度网络中，mlp也可以用，但是mlp没那么深。


# 残差网络（ResNet）由 VGG过来的
# 模型偏差：越大模型可能学习的离最优点越远；
# ResNet: f(x) = x + g(x)


import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 测试
blk = Residual(3, 3)
X = torch.rand(size=(4, 3, 6, 6))
Y = blk(X)
print(Y.shape)

# 增加通道数；输入输出长度减半
blk = Residual(3, 6, use_1x1conv=True, stride=2)
Y = blk(X)
print(Y.shape)


# ResNet-18模型
# 类似GoogleNet b1 增加了BN
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, output_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(output_channels, output_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

# 测试
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# loss 0.011, train acc 0.998, test acc 0.911
# 4259.7 examples/sec on cuda:0

# residual怎么处理梯度消失的
# residual梯度不会消失.png


# TODO kaggle竞赛：Classify Leaves
# TODO 课程31-50
