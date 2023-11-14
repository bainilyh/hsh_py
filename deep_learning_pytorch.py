import torch

print(torch.__version__)
# 创建数组 （形状、类型、值）
x = torch.zeros(2, 3, 4)
x = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]])
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.arange(12)

x = x.reshape(3, 4)

# 运算 + - * / **
# 张量链接 cat
x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
y = torch.tensor([[4.0, 3, 1], [1, 2, 3], [3, 3, 2], [1.0, 2.0, 4]])
print(torch.cat((x, y), dim=0))
print('*' * 7)
print(torch.cat((x, y), dim=1))

# 逻辑运算构建张量
a = x == y
print(a)

# 求和
print(x.sum())

# 广播机制 前提确保维度一样
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a.dim(), b.dim(), a.dim() == b.dim())
print(a, b, sep='\n')
print(a + b)

# 元素的访问 原地操作
print(x)
x[0:2, :] = 100
print(x)

z = torch.zeros_like(x)
tmp = id(z)
z[:] = x + y
print(tmp, id(z), tmp == id(z), sep='\t')

# 转换为numpy
a = x.numpy()
b = torch.tensor(a)
print(a, b, type(a), type(b), sep='\n')

# 大小为1的张量转python标量
x = torch.tensor([1.0])
print(x, x.item(), int(x), float(x))

# reshape 只是改变形状
a = torch.arange(12)
b = a.reshape(3, 4)
b[:] = 1
print(a, 'id(a) = {0}, id(b) = {1}'.format(id(a), id(b)), sep='\n')

# 矩阵计算最主要是对矩阵求导数
# 自动求导:计算一个函数在指定值上的导数；区别于（符号求导；数值求导）
# 反向传播:缓存前向的值；一层一层求导；
# 计算y=2 * x**2的导数
x = torch.arange(4, dtype=torch.float32, requires_grad=True)
# 存梯度
# x.requires_grad_(True)
print(x.grad)
y = 2 * torch.matmul(x, x)  # 2 * torch.dot(x, x)
print(y)
# 调用反向传播计算y对于x每个分量的梯度
y.backward()
print(x.grad, x.grad == 4 * x)

# 另一个例子
x = torch.randn(size=(), requires_grad=True)
print('x = {}'.format(x.item()))
y = 2 * x ** 2
y.backward()
print(x.grad, 'x.grad == 4 * x ? {}'.format(x.grad.item() == 4 * x.item()))

# 清空梯度
x.grad.zero_()
y = x.sum()  # 向量的sum梯度是全1
y.backward()
print(x.grad)

# 深度学习很少对向量/矩阵求导，而是对标量求导；所以遇到对向量/矩阵求导先sum()再反向传播
x.grad.zero_()
y = x * x
print(x, y, sep='\n')
y.sum().backward()
print(x.grad)

# 将计算移动到计算图之外
x.grad.zero_()
y = x * x
u = y.detach()  # 分离；不更新梯度计算；（固定参数）
z = u * x
z.sum().backward()
print(x.grad == u)

# TODO 控制流计算

# 线性回归
# 模型；损失函数；训练数据（学习参数）；优化方法（梯度下降）
# 优化方法超参数：学习率；批量大小

# 线性回归从零实现
import random
import torch
from d2l import torch as d2l


# 生成数据
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪音"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])
features, labels = synthetic_data(true_w, true_b, 100000)


# 批量读取
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 1000
for X, y in data_iter(batch_size, features, labels):
    print(X, y, sep='\n')
    break

# 定义模型和参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(size=1, requires_grad=True)


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数（均方误差）
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):  # lr是学习率
    """小批量随机梯度下降"""
    with torch.no_grad():  # 更新参数时候不需要计算梯度
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 组合代码 train()
lr = 0.03
num_epoch = 3
net = linreg
loss = squared_loss

for epoch in range(num_epoch):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()  # 由于batch存在l是个向量
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_loss = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

# 线性回归的简洁实现
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

print(torch.__version__)

# 构造数据
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([-1])
features, labels = d2l.synthetic_data(true_w, true_b, 10000)


# 批量读数据
def load_array(datas, batch_size, is_train=True):
    """构造一个torch数据迭代器"""
    dataset = data.TensorDataset(*datas)  # 将tensor转成TensorDataset对象
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 通过TensorDataset对象构建Dataloader


batch_size = 100
data_iter = load_array((features, labels), batch_size)

# 模型定义和初始化参数
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

# net[0].weight是Parameter对象，是torch.Tensor的子类；属性data获取tensor对象
# 同理net[0].bias是Parameter对象
# net[0].weight.data.normal_(0, 0.01)
net[0].weight.data.fill_(0)
net[0].bias.data.fill_(0)

# 损失函数
loss = nn.MSELoss()  # 均方误差

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epoch = 3

for epoch in range(num_epoch):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    train_loss = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {train_loss:f}')

# softmax
# MNIST手写数字识别；ImageNet自然物体分类（1000类）
# 交叉熵用来衡量两个概率的区别 H(p, q) = sigma(-pi * log(qi))
# 用交叉熵做损失 l(y, y_hat) = - sigma(yi * log(y_hati)) = - simga(log(y_hati))
# softmax导数:真实概率与预测概率的区别 softmax(o)i - yi

# 损失函数：衡量预测值和真实值之间的区别
# L2 Loss: 1/2 * (y - y_hat)**2
# L1 Loss: abs(y - y_hat)


# MNIST 图像分类
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 图像数据处理
# 图像是PIL类型
trans = transforms.ToTensor()  # 将图像转换为32位浮点tensor
# 读取数据
mnist_train = torchvision.datasets.FashionMNIST(
    root='./data', train=True, transform=trans, download=True
)

mnist_test = torchvision.datasets.FashionMNIST(
    root='./data', train=False, transform=trans, download=True
)

print(len(mnist_train), len(mnist_test))
print(type(mnist_train[0][0]))

# # 可视化数据集
# def get_fashion_mnist_labels(labels):
#     """返回Fashion-MNIST数据集的⽂本标签"""
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]

# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
#     """绘制图像列表"""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             # 图⽚张量
#             ax.imshow(img.numpy())
#         else:
#             # PIL图⽚
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# 构建批量数据
batch_size = 256


def get_dataloader_wokers():
    """使用4个进程来读数据"""
    return 4


# mnist_train是torch.utils.data.Dataset子类
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_wokers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop(): 2f} sec')


# 整合
def load_data_fashion_mnist(batch_size):
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)  # torchvision.transforms.Compose组合转换策略

    mnist_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_wokers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_wokers()))


# softmax回归从零开始
import torch
from torch import nn
from d2l import torch as d2l

# 定义数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

# 定义参数
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 实现softmax
# 两个维度sum
a = torch.tensor([[3.0, 1.0, 4], [1, 2.0, 3.0]])
print(a.sum(dim=0, keepdim=True))
print(a.sum(dim=1, keepdim=True))


def softmax(X):  # X是个矩阵了
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)  # 使用广播机制（dim维度一定得保持一致）
    return X_exp / partition


# 测试
X = torch.normal(0, 1, size=(2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))


# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)


# 定义损失函数（交叉熵）
# 取出tensor中目标索引上的概率
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])  # 取y_hat0行样本，索引是0的值;取y_hat1行样本，索引值是2的值


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


print(cross_entropy(y_hat, y))


# 定义度量指标
def accuracy(y_hat, y):
    '''计算预测正确的数量'''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 返回最大值的索引
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


print(accuracy(y_hat, y) / len(y))


class Accumulator:
    '''存放分类正确的个数和总的样本数'''

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *arg):
        self.data = [a + float(b) for a, b in zip(self.data, arg)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


a = Accumulator(2)
a.add(1, 3)
a.add(2, 7)
a.add(2, 10)
print(a[0] / a[1])


def evaluate_accuracy(net, data_iter):
    '''计算整个数据集上的精度'''
    if isinstance(net, nn.Module):
        net.eval()  # 进入评估模式，不计算梯度
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # y.numel()y的个数
    return metric[0] / metric[1]


print(evaluate_accuracy(net, test_iter))


# 整合
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 训练模式（更新梯度）
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l) * len(y),
                accuracy(y_hat, y),
                y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(
                float(l.sum()),
                accuracy(y_hat, y),
                y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    # TODO 显示动画
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1} train_loss {train_metrics[0]} train_acc {train_metrics[1]} test_acc {test_acc}')


# 定义优化器
lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


# 开始训练
num_epoch = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epoch, updater)

# TODO 开始预测

# softmax回归的简洁实现
import torch
from torch import nn
from d2l import torch as d2l

# 定义数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义模型、初始化参数
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # nn.init


net.apply(init_weights)

# 定义交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 设置优化器 lr设置0.1
trainer = torch.optim.SGD(net.parameters(), lr=0.1)  # net.parameters()获取模型参数

# 训练
num_epochs = 10
# 显示图片有问题
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# 问题
# 1.使用softlabel策略弥补sotfmax无法拟合0和1的值


# 多层感知机（全连接）
# 超参数：隐藏层数；每层隐藏层的大小
# 神经网络的本质是信息压缩的过程，所以隐藏层大小越往后越小比较合理。
# 常用激活函数Sigmoid,Tanh,ReLU；使用softmax处理多分类问题。

import torch
from torch import nn
from d2l import torch as d2l

# 定义数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
param = [W1, b1, W2, b2]


# 定义激活函数
def relu(X):
    A = torch.zeros_like(X)
    return torch.max(X, A)


# 定义模型
def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X @ W1 + b1)  # torch.matmul(X, W1)
    return H @ W2 + b2


# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化器
lr = 0.1
updater = torch.optim.SGD(param, lr=lr)

# 开始训练
num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 多层感知机的简洁实现
import torch
from torch import nn
from d2l import torch as d2l

# 定义数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义模型；初始化参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_inputs, num_hiddens),
                    nn.ReLU(),
                    nn.Linear(num_hiddens, num_outputs))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)


net.apply(init_weights)

# 定义损失函数
loss = nn.CrossEntropyLoss()
# 定义优化器
lr = 0.1
updater = torch.optim.SGD(net.parameters(), lr=lr)
# 训练
num_epochs = 3
# train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 模型选择；训练误差；泛化误差；验证数据集；测试数据集
# 上述的测试数据集其实是验证数据集；测试数据集不能用来调参
# K则交叉验证

# 过拟合和欠拟合
# 估计模型容量：参数个数；参数值范围。
# VC维 TODO 深度学习很少用
# 数据复杂度：样本个数；样本维度；时间、空间结构；（类别）多样性

# TODO 多项式回归的例子
# TODO 课后问答


# 权重衰退（处理过拟合）
# 增加限制：||w||**2 < theta；使用均方范数作为柔性限制。
# 在进行权重更新的时候w(t+1) = w(t) - 系数1 * 梯度的反方向，增加限制后是
# w(t+1) = (1 - 系数1*系数2) * w(t) - 系数1 * 梯度的反方向；通常系数1*系数2是小于1的，所以称为权重衰减。

# 代码实现
import torch
from torch import nn
from d2l import torch as d2l

# y=0.05 + sigma(0.01x(i)) + 噪音
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


# 从0开始实现
# 定义参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义l2惩罚项
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def l1_penalty(w):
    return torch.abs(w)


# 定义训练函数（包含模型等）
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss  # 定义模型和损失函数
    num_epochs, lr = 100, 0.03  # 定义迭代次数和步长
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)  # 广播机制
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


train(3.0)


# 简洁实现；主要是weight_decay参数
def train_concise(wd):
    # 模型
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    # 初始化参数
    for param in net.parameters():
        param.data.normal_()
    # 损失函数
    loss = nn.MSELoss(reduction='none')  # TODO reduction?
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([{'params': net[0].weight, 'weight_decay': wd}, {'params': net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())


train_concise(3.0)

# 课后问答
# wd一般取0.001或者0.01,一般权重衰减效果不太好。


# 丢弃法 dropout (dropout是给全连接用的，BN是给卷积用的）
# 在数据中增加噪音 == 正则（如l2正则)
# dropout在层中增加噪音；加入噪音后期望E(X') = X；
# x' = {0; x/(1 - p)} p是0到1之间的数；一定概率使得x是0，一定概率增大x

# 训练的使用位置
# h = relu(W1 * x + b1)
# h' = dropout(h) # 用在激活函数后；很少用在cnn上；
# o = W2 * h' + b2
# y = softmax(o)

# 推理不需要dropout

# 当年解释是dropout训练子神经网络，多个子神经网络要比原神经网络要好。。。
# p一般取0.1;0.5;0.9

# 实现Dropout
import torch
from torch import nn
from d2l import torch as d2l


# torch.rand(3,4)
# torch.Tensor(3,4).uniform_(0, 1)

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)

    if dropout == 0:
        return X

    # mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    mask = (torch.rand(X.shape) > dropout).float()
    # X[mask] = 0 对cpu和gpu都不好；效率不好；赋值远远没有乘法效率高。
    return mask * X / (1 - dropout)


# 测试
X = torch.arange(16, dtype=torch.float32).reshape(2, 8)
print(X)
print(dropout_layer(X, 0.0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.0))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
# 定义模型
dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.is_training = is_training
        self.lin1 = nn.Linear(self.num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs)))
        if self.is_training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.is_training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 训练和测试
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# 简洁实现
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(dropout1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout2),
    nn.Linear(num_hiddens2, num_outputs)
)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01, mean=0)


net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# 课后问答


# 数值稳定
# 向量对向量求导其结果是矩阵；当通过链式法则求导时候基本都是矩阵乘法运算
# 数值稳定常见的两个问题：梯度爆炸 1.5**100 == 4 * 10**17；梯度消失 0.8**100 == 2 * 10**(-10)。

# 让训练更加稳定
# 1.让乘法变加法（ResNet，LSTM）
# 2.梯度归一化；梯度裁剪
# 3.合理的权重初始化和激活函数


# 让每层的方差是个常数
# *将每层的输出和梯度都看作随机变量
# *让他们的均值和方差都保持一致


# 权重初始化
# 合理区间随机初始化参数；训练开始后更容易数值不稳定。
# 使用N(0, 0.01)初始化对小网络没问题，但是不能深度网络。

# Xavier初始化
# n(t)第t层的输出
# 正态分布：N(0, (2 / (n(t-1) + n(t))**1/2)
# 均匀分布：U(-(6 / (n(t-1) + n(t))**1/2, (6 / (n(t-1) + n(t))**1/2)


# 假设激活函数是线性的；激活函数不改变输入或者梯度的方差和均值的化，必须偏置位0，系数为1。
# 所以'合理'的激活函数是f(x) = x
# sigmoid(x) = 1/2 + x/4 - x**3/48 +...
# tanh(x) = 0 + x - x**3/3 + ...
# relu(x) = 0 + x for x>= 0
# 调整后的sigmoid
# 4 * sigmoid(x) - 2


# 课后回答
# 出现inf一般是lr太大或者权重初始化不对。
# 出现nan一般是/0了；或者梯度太大。


# kaggle：房价预测
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 数据预处理
# 删除Id特征，合并训练和测试集
all_features = pd.concat((train_data.iloc[:, 1: -1], test_data.iloc[:, 1:]))
# 设置数值型数据均值为0，方差为1
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 设置缺失值是为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 设置独热编码
# dummy_na=True也是一个特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# 通过.values提取numpy数据；再放入到tensor中
n_train = train_data.shape[0]
train_feature = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_feature = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# 标签数据也改成2维
train_labels = torch.tensor(train_data['SalePrice'].values.reshape(-1, 1), dtype=torch.float32)

# TODO


# 层和块
# 简单的多层感知机
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
print(net(X))


# nn.Sequential定义了特殊的nn.Module

# 自定义MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print(net(X))


# 自定义Sequential类
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # OrderedDict
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))


# 更灵活的参数和前向计算
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand(20, 20, requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1.0)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
net(X)

# 嵌套使用 不写了...


# 参数管理
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 参数访问
# Sequential类似list的对象
# 全连接层两个参数 weight,bias
print(net[2].state_dict())
print(net[2].weight)  # Parameter对象
print(net[2].weight.data)  # 权重tensor
print(net[2].weight.grad)  # 梯度

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
print(net.state_dict()['2.weight'])


# 从嵌套块中收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'blcok {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)  # 大致了解网络长什么样;用于简单网络


# 内置初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


rgnet.apply(init_weights)
print(rgnet[0][0][0].weight.data[0])


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


rgnet.apply(init_constant)
print(rgnet[0][0][0].weight.data[0])


# 对某些应用层使用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data)
print(net[2].weight.data)

# 自定义初始化 不写了...
# 直接拿出tensor做替换 不写了...


# 参数绑定 在不同的网络层中间共享权重
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
print(net(X))

print(net[2].weight.data[0], net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[:] = 1.0
print(net[2].weight.data[0], net[2].weight.data[0] == net[4].weight.data[0])

# 自定义层（和自定义网络没有本质区别）
import torch
from torch import nn
from torch.nn import functional as F


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.FloatTensor([1.0, 2.0, 3.0, 4.0, 5.0])))

# 融入Sequentail中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())  # 接近于0


# 带参数的图层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))  # 自定义Parameter类型
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


dense = MyLinear(5, 3)
print(dense.weight)

# 读写文件
import torch
from torch import nn
from torch.nn import functional as F

# 加载和保存张量
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

# 张量的list和map
y = torch.zeros(4)
torch.save([x, y], 'x-file')

x2, y2 = torch.load('x-file')
print(x2, y2)

my_dict = {'x': x, 'y': y}
torch.save(my_dict, 'x-file')

my_dict2 = torch.load('x-file')
print(my_dict2)


# 原版torch不好存网络结构；torchscript存？；只存权重
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))


net = MLP()
X = torch.randn(2, 20)
Y = net(X)
print(Y)
# 将模型的参数存储为mpl.params文件
torch.save(net.state_dict(), 'mlp.params')
# 加载mpl.params
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()  # 评估模式
print(Y == clone(X))


# 课后提问
# pd.get_dummies内存爆炸问题；
# kaming初始化和xavier初始化差不多


# GPU
import torch
from torch import nn
# cpu
torch.device('cpu')
# 第0个gpu
torch.cuda.device('cuda')
# 第1个gpu
torch.cuda.device('cuda:1')

# 查看gpu个数
print(torch.cuda.device_count())

# 兼容代码
def try_gpu(i=0):
    """如果存在，返回gpu,不能存在,返回cpu"""
    if torch.cuda.device_count() >= i + 1:
        return torch.cuda.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [torch.cuda.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_all_gpus())

# 查看张量的位置
x = torch.tensor(([1, 2, 3]))
print(x.device) # 默认在cpu上

# 创建时候放在GPU上
X = torch.ones(2, 3, device=try_gpu())
print(X)

# X + Y 得确保在同一个设备上
# 将X移植到gpu2上
# X.cuda(1)

# 神经网络
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu()) #Module.to 特有的

print(net(X))
# 查看权重的位置
print(net[0].weight.data.device)


# 课后问答
# 数据预处理可以在cpu上，也可以在gpu上




