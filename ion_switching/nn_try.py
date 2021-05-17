import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf



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
from tensorflow.keras.layers import Input, Dense, Add, BatchNormalization, Dropout
from tensorflow.keras.models import Model, Sequential
from keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# 1 slow open channel

a = 0
b = 500000
c = 500000
d = 1000000

X_train = np.concatenate([train.signal.values[a:b], train.signal.values[c:d]])
y_train = np.concatenate([train.open_channels.values[a:b], train.open_channels.values[c:d]])
one_hot_labels = to_categorical(y_train, num_classes=2)

model_1_slow_channel = Sequential()
model_1_slow_channel.add(Dense(32, activation='relu', input_dim=1))
model_1_slow_channel.add(Dropout(0.5))
model_1_slow_channel.add(Dense(2, activation='softmax', input_dim=32))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_1_slow_channel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model_1_slow_channel.fit(X_train, one_hot_labels, epochs=2)
print('Training model_1_slow_open_channel...')
preds = model_1_slow_channel.predict(X_train)
prediction = [ np.argmax(x) for x in preds ]
print('has f1 validation score =', f1_score(y_train, prediction, average='macro'))

# 1 fast open channel

a = 1000000
b = 1500000
c = 3000000
d = 3500000

X_train = np.concatenate([train.signal.values[a:b], train.signal.values[c:d]])
y_train = np.concatenate([train.open_channels.values[a:b], train.open_channels.values[c:d]])
one_hot_labels = to_categorical(y_train, num_classes=2)

model_1_fast_channel = Sequential()
model_1_fast_channel.add(Dense(32, activation='relu', input_dim=1))
model_1_fast_channel.add(Dropout(0.5))
model_1_fast_channel.add(Dense(2, activation='softmax', input_dim=32))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_1_fast_channel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model_1_fast_channel.fit(X_train, one_hot_labels, epochs=2)
print('Training model_1_fast_open_channel...')
preds = model_1_fast_channel.predict(X_train)
prediction = [np.argmax(x) for x in preds ]
print('has f1 validation score =', f1_score(y_train, prediction, average='macro'))

# 3 open channels

a = 1500000
b = 2000000
c = 3500000
d = 4000000

X_train = np.concatenate([train.signal.values[a:b], train.signal.values[c:d]])
y_train = np.concatenate([train.open_channels.values[a:b], train.open_channels.values[c:d]])
one_hot_labels = to_categorical(y_train, num_classes=4)

model_3_channels = Sequential()
model_3_channels.add(Dense(32, activation='relu', input_dim=1))
model_3_channels.add(Dropout(0.5))
model_3_channels.add(Dense(12, activation='relu', input_dim=32))
model_3_channels.add(Dense(4, activation='softmax', input_dim=12))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_3_channels.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model_3_channels.fit(X_train, one_hot_labels, epochs=2)
print('Training model_3_open_channel...')
preds = model_3_channels.predict(X_train)
prediction = [ np.argmax(x) for x in preds ]
print('has f1 validation score =', f1_score(y_train, prediction, average='macro'))

# 5 open channels

a = 2500000
b = 3000000
c = 4000000
d = 4500000


X_train = np.concatenate([train.signal.values[a:b], train.signal.values[c:d]])
y_train = np.concatenate([train.open_channels.values[a:b], train.open_channels.values[c:d]])
one_hot_labels = to_categorical(y_train, num_classes=7)

model_5_channels = Sequential()
model_5_channels.add(Dense(32, activation='relu', input_dim=1))
model_5_channels.add(Dropout(0.5))
model_5_channels.add(Dense(7, activation='softmax', input_dim=32))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_5_channels.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model_5_channels.fit(X_train, one_hot_labels, epochs=2)
print('Training model_5_open_channel...')
preds = model_5_channels.predict(X_train)
prediction = [ np.argmax(x) for x in preds ]
print('has f1 validation score =', f1_score(y_train, prediction, average='macro'))

# 10 open channels

a = 2000000
b = 2500000
c = 4500000
d = 5000000

X_train = np.concatenate([train.signal.values[a:b], train.signal.values[c:d]])
y_train = np.concatenate([train.open_channels.values[a:b], train.open_channels.values[c:d]])
one_hot_labels = to_categorical(y_train, num_classes=11)

model_10_channels = Sequential()
model_10_channels.add(Dense(32, activation='relu', input_dim=1))
model_10_channels.add(Dropout(0.5))
model_10_channels.add(Dense(11, activation='softmax', input_dim=32))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_10_channels.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model_10_channels.fit(X_train, one_hot_labels, epochs=3)
print('Training model_10_open_channel...')
preds = model_10_channels.predict(X_train)
prediction = [np.argmax(x) for x in preds ]
print(set(prediction))
print('has f1 validation score =', f1_score(y_train, prediction, average='macro'))


sub = pd.read_csv('./data/sample_submission.csv')
a = 100000
result = []

# part 1
predict = model_1_slow_channel.predict(test.signal.values[0*a:1*a])
result += [np.argmax(x) for x in predict]
# part 2
predict = model_3_channels.predict(test.signal.values[1*a:2*a])
result += [np.argmax(x) for x in predict]
# part 3
predict = model_5_channels.predict(test.signal.values[2*a:3*a])
result += [np.argmax(x) for x in predict]
# part 4
predict = model_1_slow_channel.predict(test.signal.values[3*a:4*a])
result += [np.argmax(x) for x in predict]
# part 5
predict = model_1_fast_channel.predict(test.signal.values[4*a:5*a])
result += [np.argmax(x) for x in predict]
# part 6
predict = model_10_channels.predict(test.signal.values[5*a:6*a])
result += [np.argmax(x) for x in predict]
# part 7
predict = model_5_channels.predict(test.signal.values[6*a:7*a])
result += [np.argmax(x) for x in predict]
# part 8
predict = model_10_channels.predict(test.signal.values[7*a:8*a])
result += [np.argmax(x) for x in predict]
# part 9
predict = model_1_slow_channel.predict(test.signal.values[8*a:9*a])
result += [np.argmax(x) for x in predict]
# part 10
predict = model_3_channels.predict(test.signal.values[9*a:10*a])
result += [np.argmax(x) for x in predict]
# part 11
predict = model_1_slow_channel.predict(test.signal.values[10*a:20*a])
result += [np.argmax(x) for x in predict]


sub['open_channels'] = np.array(result)
print("training successful!")


plt.figure(figsize=(20,5))
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


sub.to_csv('submission.csv', index = False, float_format='%.4f')
print("submission.csv saved successfully!")