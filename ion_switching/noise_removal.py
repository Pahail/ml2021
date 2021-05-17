import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, fftfreq, ifft



train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')




def total_fft(df, sig='signal', limits=(-100, 100), logy=False, d=1e-4):
    sig_fft = fft(df[sig].values)
    power = np.abs(sig_fft)
    freq = fftfreq(df.shape[0], d=d)
    plt.figure(figsize=(12,4))
    l, r = limits
    mask = np.where((l <= freq) & (freq <= r))
    plt.plot(freq[mask], power[mask])
    plt.grid()
    if logy:
        plt.yscale('log')
    plt.show()


total_fft(train)
