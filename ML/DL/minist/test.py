import csv
import pickle
import numpy as np
from matplotlib import pyplot as plt


def diplay_test():	# 读取测试集，预测，画图
    file_name = 'data/test.csv'
    file = open('NN.txt', 'rb')
    nn = pickle.load(file)
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        header_row = next(reader)
        print(header_row)
        i = 0
        for row in reader:
            i += 1
            img = np.array(row, dtype=np.uint8)
            img = img.reshape(28, 28)
            plt.imshow(img, cmap='gray')
            pre, lst = nn.predict(row)
            plt.title(str(pre), fontsize=24)
            plt.axis('off')
            plt.savefig('img/img' + str(i) + '.png')


diplay_test()

