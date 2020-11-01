from main import NeuralNetwork
import numpy as np
import pickle
import csv


def train():
    file_name = '../minist/data/train.csv'	# 数据集为42000张带标签的28x28手写数字图像
    y = []
    x = []
    y_t = []
    x_t = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        header_row = next(reader)
        print(header_row)
        for row in reader:
            if np.random.random() < 0.8:	# 大约80%的数据用于训练
                y.append(int(row[0]))
                x.append(list(map(int, row[1:])))
            else:
                y_t.append(int(row[0]))
                x_t.append(list(map(int, row[1:])))
    len_train = len(y)
    len_test = len(y_t)
    print('训练集大小%d，测试集大小%d' % (len_train, len_test))
    x = np.array(x)
    y = np.array(y)
    nn = NeuralNetwork([784, 784, 10])	# 神经网络各层神经元个数
    nn.fit(x, y)
    file = open('NN.txt', 'wb')
    pickle.dump(nn, file)
    count = 0
    for i in range(len_test):
        p, _ = nn.predict(x_t[i])
        if p == y_t[i]:
            count += 1
    print('模型识别正确率：', count/len_test)


def mini_test():	# 小型测试，验证神经网络能正常运行
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 2, 3]
    nn = NeuralNetwork([2, 4, 16, 4])
    nn.fit(x, y, epochs=10000)
    for i in x:
        print(nn.predict(i))


# mini_test()
train()
