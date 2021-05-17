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

X_train = np.zeros((train.signal.values.shape[0], 2))
X_train[:, 0] = train.signal.values
X_train[:-1, 1] = train.signal.values[1:]
y_train = np.array(train.open_channels.values)
one_hot_labels = to_categorical(y_train, num_classes=11)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=2))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', input_dim=1))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', input_dim=1))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax', input_dim=32))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, one_hot_labels, epochs=3)
print('Training model...')
preds = model.predict(X_train)
prediction = [ np.argmax(x) for x in preds ]
print('has f1 validation score =', f1_score(y_train, prediction, average='macro'))


X_test = np.zeros((test.signal.values.shape[0], 2))
X_test[:, 0] = test.signal.values
X_test[:-1, 1] = test.signal.values[1:]

sub = pd.read_csv('./data/sample_submission.csv')
result = [np.argmax(x) for x in model.predict(X_test)]

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