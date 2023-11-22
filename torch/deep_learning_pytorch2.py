# 序列模型
# 联合概率可以用条件概率展开
# x(t)为t时刻的事件，一共T个事件
# p(X) = p(x1, x2,...,xT) = p(x1) * p(x2|x1) * p(x3|x1, x2)...p(xT|x1,..x(T-1))
# = P(xT) * P(x(T-1)|x(T)) * .... * p(x1|x2,..,xT)

# 核心算：p(xt|x1,...,x(t-1))；可以对x1,..,x(t-1)建模 -> f(x1,..,x(t-1))
# 这里的f是自回归模型；标号和数据是一起的。

# A.马尔可夫假设：当前数据只跟过去tau个数据相关。
# p(xt|x1,..,x(t-1)) = p(xt|x(t-tau),..,x(t-1)) = p(xt|f(x(t-tau),..,x(t-1)))
# 这里f是可以用MLP模型进行训练；因为是定长的数据。

# B.潜变量（隐变量的推广）模型：RNN
# 引入潜变量ht表示过去信息，ht = f(x1,..,x(t-1)), xt = p(xt|ht);ht是个向量或者一个数。
# 潜变量模型.png

# %%
# 测试：正弦函数+噪音，T个时间点数据
import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# 图像
d2l.plot(time, [x], 'time', 'x', xlim=(1, 1000), figsize=(6, 3))
d2l.plt.show()

# %%
# 使用马尔可夫假设构建模型 tau=4
# 1.将数据映射为数据对
tau = 4
features = torch.zeros(T - tau, tau)
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
label = x[tau:].reshape(-1, 1)

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], label[:n_train]), batch_size, is_train=True)


# 构造模型和参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


loss = nn.MSELoss()


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}', f'loss: {d2l.evaluate_loss(net, train_iter, loss)}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 单步预测（预测一个点）
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()],
         'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# 多步预测？（预测多个点）TODO 完全不准
# 接下里模型的发展是尽可能多的预测未来...

# %%
# 文本预处理；把文本当作时序序列（整个MLP在干的事）
import collections
import re
from d2l import torch as d2l


def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 把不是A-Za-z的字符变成空格
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()
print(f'文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])


# 把一行文本序列变成token序列；token有两种表示：词和字符。
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('未知token: ', token)


tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


# 词汇表；把token(word/char)映射成从0开始的数字索引
class Vocab:
    """⽂本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    # 这⾥的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成⼀个列表
        tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)


# 构建词汇表
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
print(list(vocab.idx_to_token[:10]))

# 测试，将0和10行的token转成数字
for i in [0, 10]:
    print('words', tokens[i])
    print('indices', vocab[tokens[i]])


# 整合功能
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，
    # 所以将所有⽂本⾏展平到⼀个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))

# 课后问答
# 为啥要对词典进行排序，经常访问的token放到一块，有利于计算机访问，加快速度。


# %%
# 语言模型；给点x1,...,xT,语言模型的目标是估计联合概率p(x1,..,xT)
# 应用：BERT，GPT3等预训练模型；生成文本；判断多个序列种哪个更常见

# 使用计数来建模；使用计数来建模.png
# N元语法；N元语法.png；马尔可夫模型中tau的个数

# 语言模型和数据集
import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])
# [('the', 2261), ('i', 1267), ('and', 1245), ('of', 1155), ('a', 816), ('to', 695), ('was', 552), ('in', 541), ('that', 443), ('my', 440)]
# 这些常见词就是stop word（停用词） 虚词

# 词元组合，比如二元，三元
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])


# 三元...

# 画词频 TODO ..
# 虽然ngram是指数级别的增加，但是现实场景用的也比较多。


# 将corpus -> mini batch
# A.随机采样
def seq_data_iter_random(corpus, batch_size, num_steps):  # num_stemps相当于tau
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


# 测试生成一个0到34的序列的mini_batch
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


# B.顺序分区 TODO
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使⽤顺序分区⽣成⼀个⼩批量⼦序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


# 整合
class SeqDataLoader:
    """加载序列数据的迭代器"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# 最终函数：返回迭代器和词表
def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# %%
# 潜变量自回归模型：潜变量模型.png
# p(ht|h(t-1),x(t-1))
# p(xt|ht, x(t-1))
# RNN;隐变量ht是向量;x(t-1) ->(输出) ht ->(输出) ot;计算损失是比较ot和xt之间的损失
# 输入:你好，世界！
# 输入你，更新隐变量，预测好。输入好，更新隐变量，预测，....
# RNN.png；和全连接比较，多了啥？

# 衡量语言模型好坏；分类问题 -> 交叉熵
# pi = 1 / n * sigma(-logP(xt|x(t-1),..))
# 历史原因：使用困惑度衡量:exp(pi)

# 梯度裁剪：计算T个时间步的梯度；会有O(T)词矩阵乘法链；导致数值不稳定；
# 梯度裁剪有效防止梯度爆炸；操作梯度长度

# 应用：文本生成；文本分类；问答、机器翻译；Tag生成


# 课后问答
# %%
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 独热编码
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

# 小批量数据形状（批量大小，时间步数）
X = torch.arange(10).reshape(2, 5)
print(F.one_hot(X.T, 28).shape)  # 按时间步数批量迭代


# 初始化参数;RNN.png
def get_params(vocab_size, num_hiddens, device):
    # 输入输出都是一个词的emb或者one_hot
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))  # 比MLP多了这行
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出ot
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 初始化时候返回隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    # 返回tuple将来LSTM需要
    return (torch.zeros((batch_size, num_hiddens), device=device),)

# 定义rnn函数
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs: #时间步进行迭代
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, ) # 1.行数是批量大小*时间步长 2. H用于下次

# 类包装函数
class RNNModelScratch:
    """从零实现循环神经网络"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 初始化RNN
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape, sep='; ')


# 预测code
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda : torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    # 预热
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 预测
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[id] for id in outputs])

# 测试predict_ch8 prefix='time traveller'
print(predict_ch8('time traveller', 10, net, vocab, d2l.try_gpu()))

# 梯度裁剪 TODO
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [param for param in net.parameters() if param.requires_grad]
    else:
        params = net.params
    # 获取所有梯度的norm
    norm = torch.sqrt(sum(torch.sum((param.grad ** 2)) for param in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# 一个周期内的迭代code
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) #损失之和，词元数量
    for X, y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(X.shape[0], device=device)
        else: # TODO!!!
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        y = y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# 训练函数
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), use_random_iter=True)


# %%
# 简洁实现
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 数据
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 模型
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 初始化隐藏层权重；比从零实现多了个隐藏层数；（隐藏层数，批量大小，隐藏单元数）
state = torch.zeros(size=(1, batch_size, num_hiddens))

# 测试
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
# 隐状态的输出；相当于ot
print(Y.shape, state_new.shape) # nn.RNN的输出；不涉及输出层的计算：它是指每个时间步的隐状态，这些隐状态可以⽤作后续输出层的输⼊
# torch.Size([35, 32, 256]) torch.Size([1, 32, 256])

# 模型
class RNNModel(nn.Module):
    """循环神经⽹络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层⾸先将Y的形状改为(时间步数*批量⼤⼩,隐藏单元数)
        # 它的输出形状是(时间步数*批量⼤⼩,词表⼤⼩)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers,batch_size, self.num_hiddens), device=device),
                    torch.zeros(( self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

# 测试
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)

# 训练
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
# perplexity 1.4, 280515.7 tokens/sec on cuda:0
# time traveller held in his hand war peangh wfocee ware istelatre
# traveller he had sner ared at suralexrscinctions of ipaci a

# perplexity 1.3, 56806.3 tokens/sec on cpu
# time traveller proceeded anyrean to becinturee aid the very youn
# traveller priceftityol said the time travellerit s against


# 课后问答
# torchserve 模型服务
# 1.集成到web框架；模型的推理速度受到web框架的限制，时延较高，且并发性不高，不能满足其工业化落地的标准。
# 2.将模型推理过程封装成服务，内部实现模型加载、模型版本管理、批处理以及服务接口封装等功能，对外提供RPC/HTTP接口。

# rnn记住100之内的序列还行


# %%
# GRU（门控循环单元）；是LSTM后提出来的...
# 对隐藏层的向量进行处理；关注之前重要的状态（单词）等
# 更新门；重置门
# 门.png；候选隐状态.png；总结.png； 隐状态.png

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 数据
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    # 输入输出都是一个词的emb或者one_hot
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    # W_xh = normal((num_inputs, num_hiddens))
    # W_hh = normal((num_hiddens, num_hiddens))  # 比MLP多了这行
    # b_h = torch.zeros(num_hiddens, device=device)
    W_xz, W_hz, b_z = three() # z gate
    W_xr, W_hr, b_r = three() # r gate
    W_xh, W_hh, b_h = three() # RNN
    # 输出ot
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 初始化时候返回隐藏状态
def init_gru_state(batch_size, num_hiddens, device):
    # 返回tuple将来LSTM需要
    return (torch.zeros((batch_size, num_hiddens), device=device),)

# 定义gru函数
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs: #时间步进行迭代
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z)* H_tilda
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, ) # 1.行数是批量大小*时间步长 2. H用于下次

# 训练
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(vocab_size, num_hiddens, device, get_params, init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# perplexity 1.1, 31689.6 tokens/sec on cuda:0
# time traveller with a slight accession ofcheerfulness really thi
# travelleryou can show black is white by argument said filby

# perplexity 1.1, 20673.9 tokens/sec on cpu
# time traveller but now you begin to seethe object of my investig
# traveller a megheng tis explanou acone from the trammels ge

# %%
# 简洁实现
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
num_inputs = vocab_size

gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# perplexity 1.0, 328117.7 tokens/sec on cuda:0
# time traveller for so it will be convenient to speak of himwas e
# travelleryou can show black is white by argument said filby

# perplexity 1.0, 5110.9 tokens/sec on cpu
# time traveller for so it will be convenient to speak of himwas e
# travelleryou can show black is white by argument said filby


# 课后问答
# grad clipping 一般是1、50等


# %%
# LSTM(长短期记忆网络)
# 忘记门 Ft；输入门 It；输出门 Ot
# LSTM_门.png；LSTM_候选记忆单元.png；LSTM_记忆单元.png；LSTM_隐藏状态.png；LSTM_总结.png
# 比GRU多了个记忆单元C；通过F和I来更新C

import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    # 输入输出都是一个词的emb或者one_hot
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    # W_xh = normal((num_inputs, num_hiddens))
    # W_hh = normal((num_hiddens, num_hiddens))  # 比MLP多了这行
    # b_h = torch.zeros(num_hiddens, device=device)
    # W_xz, W_hz, b_z = three() # z gate
    # W_xr, W_hr, b_r = three() # r gate
    # W_xh, W_hh, b_h = three() # RNN
    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()
    # 输出ot
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化 h 和 c
def init_lstm_state(batch_size, num_hiddens, device):
    # 返回tuple将来LSTM需要
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    for X in inputs: #时间步进行迭代
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C) # 1.行数是批量大小*时间步长 2. H用于下次

# 训练
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(vocab_size, num_hiddens, device, get_params, init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# perplexity 1.1, 26153.1 tokens/sec on cuda:0
# time travelleryou can show black is white by argument said filby
# travellerit s against reason said filbywere whine freme as

# perplexity 1.1, 12874.1 tokens/sec on cpu
# time traveller for so it will be convenient to speak of himwas e
# travelleryou can show black is white by argument said filby

# %%
# 简洁实现
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
num_inputs = vocab_size

lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# perplexity 1.3, 314164.3 tokens/sec on cuda:0
# time traveller filed and they langled of the brifilepsomy inseli
# travelleryou can show black is whetery ucad spale they soul


# 课后问答
# lstm想让h在（-1，1）之间，防止梯度爆炸


# %%
# 深度循环神经网络
# 更深02.png；更深01.png
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# num_layers确定隐藏层的深度
vocab_size, num_hiddens, device, num_layers = len(vocab), 256, d2l.try_gpu(), 2
num_epochs, lr = 500, 2
num_inputs = vocab_size

lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

# perplexity 1.0, 198800.1 tokens/sec on cuda:0
# time traveller for so it will be convenient to speak of himwas e
# travelleryou can show black is white by argument said filby

# 一般和MLP差不多，用2层最多了。