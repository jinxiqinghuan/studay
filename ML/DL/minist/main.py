import numpy as np
from tqdm import trange	# 替换range()可实现动态进度条，可忽略


def sigmoid(x): # 激活函数采用Sigmoid
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):	# Sigmoid的导数
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:	# 神经网络
    def __init__(self, layers):	# layers为神经元个数列表
        self.activation = sigmoid	# 激活函数
        self.activation_deriv = sigmoid_derivative	# 激活函数导数
        self.weights = []	# 权重列表
        self.bias = []	# 偏置列表
        for i in range(1, len(layers)):	# 正态分布初始化
            self.weights.append(np.random.randn(layers[i-1], layers[i]))
            self.bias.append(np.random.randn(layers[i]))

    def fit(self, x, y, learning_rate=0.2, epochs=3):	# 反向传播算法
        x = np.atleast_2d(x)
        n = len(y)	# 样本数
        p = max(n, epochs)	# 样本过少时根据epochs减半学习率
        y = np.array(y)

        for k in trange(epochs * n):	# 带进度条的训练过程
            if (k+1) % p == 0:
                learning_rate *= 0.5	# 每训练完一代样本减半学习率
            a = [x[k % n]]	# 保存各层激活值的列表
            # 正向传播开始
            for lay in range(len(self.weights)):
                a.append(self.activation(np.dot(a[lay], self.weights[lay]) + self.bias[lay]))
            # 反向传播开始
            label = np.zeros(a[-1].shape)
            label[y[k % n]] = 1	# 根据类号生成标签
            error = label - a[-1]	# 误差值
            deltas = [error * self.activation_deriv(a[-1])]	# 保存各层误差值的列表

            layer_num = len(a) - 2	# 导数第二层开始
            for j in range(layer_num, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))	# 误差的反向传播
            deltas.reverse()
            for i in range(len(self.weights)):	# 正向更新权值
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
                self.bias[i] += learning_rate * deltas[i]

    def predict(self, x):	# 预测
        a = np.array(x, dtype=np.float)
        for lay in range(0, len(self.weights)):	# 正向传播
            a = self.activation(np.dot(a, self.weights[lay]) + self.bias[lay])
        a = list(100 * a/sum(a))	# 改为百分比显示
        i = a.index(max(a))	# 预测值
        per = []	# 各类的置信程度
        for num in a:
            per.append(str(round(num, 2))+'%')
        return i, per

