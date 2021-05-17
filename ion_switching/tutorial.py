import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns


def print_files(path='./data'):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def test_null_values(table):
    for i in table.columns:
        # print(i, table[i].isnull().sum())
        assert table[i].isnull().sum() == 0


'''
train содержит 5 000 000 записей формата: момент времени, величина сигнала, открытые каналы
test содержит 2 000 000 записей: момент времени и сигнал 
'''
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

test_original = test.copy()

test_null_values(train)
test_null_values(test)

# Смотрим распределение открытий каналов
print(train.open_channels.value_counts())

# Смотрим корреляцию сигнала и времени
corr_dataframe = train[["time", "signal"]]
corr_mat = corr_dataframe.corr()
print(corr_mat)

plt.figure(figsize=(12, 10))
axes = plt.subplot(3, 1, 1)
plt.plot(train.time[::100], train.signal[::100])
axes.set_ylabel("Signal")
a = 500000
dist = 100
axes = plt.subplot(3, 1, 2)
axes.set_ylabel("Signal")
axes = plt.subplot(3, 1, 3)
axes.set_ylabel("Open channels")
for i in range(0, 10):
    print(i, "min: ", min(train.signal[0 + i * a:(i + 1) * a:dist].values), "max: ", max(train.signal[0 + i * a:(i + 1) * a:dist]))
    plt.subplot(3, 1, 2)
    plt.plot(train.time[0 + i * a:(i + 1) * a:dist], train.signal[0 + i * a:(i + 1) * a:dist])
    plt.subplot(3, 1, 3)
    plt.plot(train.time[0 + i * a:(i + 1) * a:dist], train.open_channels[0 + i * a:(i + 1) * a:dist])
plt.show()



'''
Как было видно из обучающих данных, на открытие каналов никак не влияет линейные участки и параболические - 
следовательно надо создать модедь, в которой они будут заменены константными участками.
'''
plt.figure(figsize=(12,6))
plt.plot(test.time[::100], test.signal[::100])
plt.show()


a = 500000
b = 600000
# plt.plot(train.time[0+1*a:(1+1)*a:dist], train.signal[0+1*a:(1+1)*a:dist])
# plt.plot(train.time[0+1*a:(1+1)*a:dist], train.open_channels[0+1*a:(1+1)*a:dist], color = 'red')
# plt.show()
#####################################
train_signal = train['signal'].copy()
train_time = train['time'].copy()
c = 0.3
d = 50
train_signal[a:b] = (train_signal[a:b].values - c * (train_time[a:b].values - d)).copy()
train['signal'] = train_signal.copy()
# plt.plot(train.time[0+1*a:(1+1)*a:dist], train.signal[0+1*a:(1+1)*a:dist])
# plt.plot(train.time[0+1*a:(1+1)*a:dist], train.open_channels[0+1*a:(1+1)*a:dist], color = 'red')
# plt.show()

################

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
################

print("correcting linear slopes in test successful!")

plt.figure(figsize=(12, 6))
plt.plot(test.time[::100], test.signal[::100])
plt.show()


def remove_parabolic_shape(values, minimum, middle, maximum):
    a = maximum - minimum
    return -(a / 625) * (values - middle) ** 2 + a


################################################

# I really want to find out, how he found these perfectly working
# numbers, because I can't imagine, that he sat around for hours,
# tweaking these low and high values until it worked.

# idea1: get the min and max value by calculating the mean
# of a certain window at the beginning of the batch
# and at the middle of the batch.


################################################
# part 7 goes from 3000k to 3500k

# his values
# low  = -1.817
# high =  3.186

# my values
# min:   -2.9517
# max:    4.366

a = 3000000
b = 3500000
minimum = -1.817
middle = 325
maximum = 3.186

train_signal = train["signal"].copy()
train_time = train["time"].copy()
train_signal[a:b] = train_signal[a:b].values - remove_parabolic_shape(train_time[a:b].values, minimum, middle, maximum)
train["signal"] = train_signal.copy()

################################################
# part 8 goes from 3500k to 4000k

# his values
# low  = -0.094
# high =  4.936

# my values
# min:   -3.0399
# max:    9.9986

a = 3500000
b = 4000000
minimum = -0.094
middle = 375
maximum = 4.936

train_signal = train["signal"].copy()
train_time = train["time"].copy()
train_signal[a:b] = train_signal[a:b].values - remove_parabolic_shape(train_time[a:b].values, minimum, middle, maximum)
train["signal"] = train_signal.copy()

################################################
# part 9 goes from 4000k to 4500k

# his values
# low  =  1.715
# high =  6.689

# my values
# min:   -2.0985
# max:    9.0889

a = 4000000
b = 4500000
minimum = 1.715
middle = 425
maximum = 6.689

train_signal = train["signal"].copy()
train_time = train["time"].copy()
train_signal[a:b] = train_signal[a:b].values - remove_parabolic_shape(train_time[a:b].values, minimum, middle, maximum)
train["signal"] = train_signal.copy()

################################################
# part10 goes from 4500k to 5000k

# his values
# low  =  3.361
# high =  8.45

# my values
# min:   -1.5457
# max:   12.683

a = 4500000
b = 5000000
minimum = 3.361
middle = 475
maximum = 8.45

train_signal = train["signal"].copy()
train_time = train["time"].copy()
train_signal[a:b] = train_signal[a:b].values - remove_parabolic_shape(train_time[a:b].values, minimum, middle, maximum)
train["signal"] = train_signal.copy()


# plt.plot(train.time, train.signal)
# plt.plot(train.time, train.open_channels, color = 'red')
# plt.show()


#######################################################
# his magical function full of magical numbers

def f(x):
    return -(0.00788)*(x-625)**2+2.345 +2.58


#######################################################

test2 = test.copy()


a = 1000000
b = 1500000

test2.signal[a:b] = test2.signal[a:b].values - f(test2.time[a:b].values)
test.signal[a:b] = test2.signal[a:b]

plt.subplot(2, 1, 1)
plt.plot(test_original.time, test_original.signal)
plt.subplot(2, 1, 2)
plt.plot(test.time, test.signal)
plt.show()
