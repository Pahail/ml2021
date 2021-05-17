import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


'''
Удаление линейныйх возрастающих участков из обучающих данных
'''

a = 500000
b = 600000
train_signal = train['signal'].copy()
train_time = train['time'].copy()
c = 0.3
d = 50
train_signal[a:b] = (train_signal[a:b].values - c * (train_time[a:b].values - d)).copy()
train['signal'] = train_signal.copy()

'''
Удаление линейныйх возрастающих участков из тестовых данных
'''

test2 = test.copy()

################
# part 1:

a = 0
b = 100000

c = 0.3
d = 500

test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)
test.signal[a:b] = test2.signal[a:b]
################
# part 2:

a = 100000
b = 200000

d = 510

test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)
test.signal[a:b] = test2.signal[a:b]
################
# part 5:

a = 400000
b = 500000

d = 540

test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)
test.signal[a:b] = test2.signal[a:b]
################
# part 7:

a = 600000
b = 700000

d = 560

test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)
test.signal[a:b] = test2.signal[a:b]
################
# part 8:

# slope  =  3/10

a = 700000
b = 800000

d = 570

test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)
test.signal[a:b] = test2.signal[a:b]
################
# part 9:

a = 800000
b = 900000

d = 580

test2.signal[a:b] = test2.signal[a:b].values - c*(test2.time[a:b].values - d)
test.signal[a:b] = test2.signal[a:b]

'''
Удаление параболических участков из обучающих данных
'''


def remove_parabolic_shape(values, minimum, middle, maximum):
    a = maximum - minimum
    return -(a / 625) * (values - middle) ** 2 + a


a = 3000000
b = 3500000
minimum = -1.817
middle = 325
maximum = 3.186

train_signal = train["signal"].copy()
train_time = train["time"].copy()
train_signal[a:b] = train_signal[a:b].values - remove_parabolic_shape(train_time[a:b].values, minimum, middle, maximum)
train["signal"] = train_signal.copy()


a = 3500000
b = 4000000
minimum = -0.094
middle = 375
maximum = 4.936

train_signal = train["signal"].copy()
train_time = train["time"].copy()
train_signal[a:b] = train_signal[a:b].values - remove_parabolic_shape(train_time[a:b].values, minimum, middle, maximum)
train["signal"] = train_signal.copy()

a = 4000000
b = 4500000
minimum = 1.715
middle = 425
maximum = 6.689

train_signal = train["signal"].copy()
train_time = train["time"].copy()
train_signal[a:b] = train_signal[a:b].values - remove_parabolic_shape(train_time[a:b].values, minimum, middle, maximum)
train["signal"] = train_signal.copy()

a = 4500000
b = 5000000
minimum = 3.361
middle = 475
maximum = 8.45

train_signal = train["signal"].copy()
train_time = train["time"].copy()
train_signal[a:b] = train_signal[a:b].values - remove_parabolic_shape(train_time[a:b].values, minimum, middle, maximum)
train["signal"] = train_signal.copy()

'''
Удаление параболических участков из тестовых данных
'''


def f(x):
    return -(0.00788)*(x-625)**2+2.345 +2.58


test2 = test.copy()

a = 1000000
b = 1500000
test2.signal[a:b] = test2.signal[a:b].values - f(test2.time[a:b].values)
test.signal[a:b] = test2.signal[a:b]


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

from data_preprocessing import build_discontinuous_model

slow_channel, fast_channel, channel_3, channel_5, channel_10 = build_discontinuous_model(train['signal'])

x_train = []
y_train = []
for s in slow_channel:
    x_train += list(train['signal'][s])
    y_train += list(train['open_channels'][s])
x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape((-1, 1))

model_1_slow_channel = tree.DecisionTreeClassifier(max_depth=1)
model_1_slow_channel.fit(x_train, y_train)
print('Training model_1_slow_open_channel...')
preds = model_1_slow_channel.predict(x_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))


x_train = []
y_train = []
for s in fast_channel:
    x_train += list(train['signal'][s])
    y_train += list(train['open_channels'][s])
x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape((-1, 1))

model_1_fast_channel = tree.DecisionTreeClassifier(max_depth=1)
model_1_fast_channel.fit(x_train, y_train)
print('Training model_1_fast_open_channel...')
preds = model_1_fast_channel.predict(x_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))


x_train = []
y_train = []
for s in channel_3:
    x_train += list(train['signal'][s])
    y_train += list(train['open_channels'][s])
x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape((-1, 1))

model_3_channels = tree.DecisionTreeClassifier(max_leaf_nodes=4)
model_3_channels.fit(x_train, y_train)
print('Training model_3_open_channels')
preds = model_3_channels.predict(x_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))


x_train = []
y_train = []
for s in channel_5:
    x_train += list(train['signal'][s])
    y_train += list(train['open_channels'][s])
x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape((-1, 1))

model_5_channels = tree.DecisionTreeClassifier(max_leaf_nodes=6)
model_5_channels.fit(x_train, y_train)
print('Training model_5_open_channels')
preds = model_5_channels.predict(x_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))


x_train = []
y_train = []
for s in channel_10:
    x_train += list(train['signal'][s])
    y_train += list(train['open_channels'][s])
x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape((-1, 1))

model_10_channels = tree.DecisionTreeClassifier(max_leaf_nodes=8)
model_10_channels.fit(x_train, y_train)
print('Training model_10_open_channels')
preds = model_10_channels.predict(x_train)
print('has f1 validation score =', f1_score(y_train, preds, average='macro'))


sub = pd.read_csv('./data/sample_submission.csv')
slow_channel, fast_channel, channel_3, channel_5, channel_10 = build_discontinuous_model(test['signal'])


a = 100000
for s in slow_channel:
    sub.iloc[s, 1] = model_1_slow_channel.predict(test.signal.values[s].reshape((-1, 1)))
for s in fast_channel:
    sub.iloc[s, 1] = model_1_fast_channel.predict(test.signal.values[s].reshape((-1, 1)))
for s in channel_3:
    sub.iloc[s, 1] = model_3_channels.predict(test.signal.values[s].reshape((-1, 1)))
for s in channel_5:
    sub.iloc[s, 1] = model_5_channels.predict(test.signal.values[2 * a:3 * a].reshape((-1, 1)))
for s in channel_10:
    sub.iloc[s, 1] = model_10_channels.predict(test.signal.values[s].reshape((-1, 1)))

print("training successful!")


plt.figure(figsize=(20, 5))
res = 1000
let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
plt.plot(range(0, test.shape[0], res), sub.open_channels[0::res])
for i in range(5):
    plt.plot([i*500000, i*500000], [-5, 12.5], 'r')
for i in range(21):
    plt.plot([i*100000, i*100000], [-5, 12.5], 'r:')
for k in range(4):
    plt.text(k*500000+250000, 10, str(k+1), size=20)
for k in range(10):
    plt.text(k*100000+40000, 7.5, let[k], size=16)
plt.title('Test Data Predictions', size=16)
plt.show()


sub.to_csv('submission.csv', index=False, float_format='%.4f')
print("submission.csv saved successfully!")
