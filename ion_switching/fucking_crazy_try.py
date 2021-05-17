import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TODO хорошо бы тут выбросы почистить
def naive_signal_transform(signal, batches=100, mini_batches=500):
    signal = signal.copy()
    batches = batches
    signal_size = signal.shape[0]
    batch_size = int(signal_size / batches)
    mini_batch = mini_batches
    mini_batch_size = int(batch_size / mini_batch)

    transformed_signal = []
    for i in range(batches):
        lower_bound = []
        upper_bound = []
        signal_batch = signal[batch_size * i: batch_size * i + batch_size]
        for j in range(mini_batch):
            signal_mini_batch = signal_batch[mini_batch_size * j: mini_batch_size * j + mini_batch_size]
            lower_bound.append(np.min(signal_mini_batch))
            upper_bound.append(np.max(signal_mini_batch))
        lower = np.median(lower_bound)
        upper = np.median(upper_bound)
        transformed_signal.append(upper - lower)

    # plt.plot(transformed_signal)
    # plt.show()

    for i in range(batches):
        signal[batch_size * i: batch_size * i + batch_size] = transformed_signal[i]

    return signal


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


train['signal'] =  naive_signal_transform(train['signal'], batches=20, mini_batches=500)
print(train.shape)
plt.plot(train['time'], train['signal'])
plt.plot(train['time'], train['open_channels'], color='red')
plt.show()


test['signal'] = naive_signal_transform(test['signal'], batches=20, mini_batches=500)
print(test.shape)
plt.plot(test['time'], test['signal'])
plt.show()


'''
Обучение на конкретных участках данных
'''

from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

