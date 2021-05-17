import numpy as np
import pandas as pd


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

    transformed_signal = np.array(transformed_signal)
    for i in range(batches):
        signal[batch_size * i: batch_size * i + batch_size] = transformed_signal[i]


    # import matplotlib.pyplot as plt
    # plt.plot(transformed_signal)
    # plt.show()

    return signal


def build_discontinuous_model(signal, number=20):

    smoothed = naive_signal_transform(signal, 100, 500)
    size = signal.shape[0]
    step = int(size / number)
    parts = np.arange(0, size, step)
    slow_channel_1 = []
    fast_channel_1 = []
    open_channel_3 = []
    open_channel_5 = []
    open_channel_10 = []
    for i in parts:
        mean = np.median(smoothed[i:i+step])
        # print(mean)
        s = slice(i, i + step)

        if mean < 1.7:
            model = '1_slow_channel'
            if len(slow_channel_1) != 0:
                if slow_channel_1[-1].stop == s.start:
                    new_slice = slice(slow_channel_1[-1].start, s.stop)
                    slow_channel_1.pop(-1)
                else:
                    new_slice = s
            else:
                new_slice = s
            slow_channel_1.append(new_slice)
        elif mean < 3.0:
            model = '1_fast_channel'
            if len(fast_channel_1) != 0:
                if fast_channel_1[-1].stop == s.start:
                    new_slice = slice(fast_channel_1[-1].start, s.stop)
                    fast_channel_1.pop(-1)
                else:
                    new_slice = s
            else:
                new_slice = s
            fast_channel_1.append(new_slice)
        elif mean < 4.0:
            model = '3_open_channel'
            if len(open_channel_3) != 0:
                if open_channel_3[-1].stop == s.start:
                    new_slice = slice(open_channel_3[-1].start, s.stop)
                    open_channel_3.pop(-1)
                else:
                    new_slice = s
            else:
                new_slice = s
            open_channel_3.append(new_slice)
        elif mean < 5.5:
            model = '5_open_channel'
            if len(open_channel_5) != 0:
                if open_channel_5[-1].stop == s.start:
                    new_slice = slice(open_channel_5[-1].start, s.stop)
                    open_channel_5.pop(-1)
                else:
                    new_slice = s
            else:
                new_slice = s
            open_channel_5.append(new_slice)
        else:
            model = '10_open_channel'
            if len(open_channel_10) != 0:
                if open_channel_10[-1].stop == s.start:
                    new_slice = slice(open_channel_10[-1].start, s.stop)
                    open_channel_10.pop(-1)
                else:
                    new_slice = s
            else:
                new_slice = s
            open_channel_10.append(new_slice)

    print(slow_channel_1)
    print(fast_channel_1)
    print(open_channel_3)
    print(open_channel_5)
    print(open_channel_10)

    return slow_channel_1, fast_channel_1, open_channel_3, open_channel_5, open_channel_10
