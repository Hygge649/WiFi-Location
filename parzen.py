import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
path = "D:/desktop/c.c/UJIndoorLoc/trainingData.csv"

#Read the Flare dataset into a list and shuffle it with the random.shuffle method.
def readfile(path):
    data=pd.read_csv(path)
    data_array = np.array(data)  #数组
    np.random.shuffle(data_array)   #shuffle, fixed seed 按行重新洗牌
    column = np.arange(data_array.shape[1]) #  .shape[0])  raw
    np.random.shuffle(column)
    return data_array  #  training set:1795



dataSet = readfile(path)
location = dataSet[:1794,520:522] #521 522 经度 纬度
data=dataSet[:1794, :200]

def gaussian_pdf(x, mu, sigma):
    prob = 1/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/2/sigma**2)
    return prob


def parzen_window_pdf(x, data, sigma):
    px = [gaussian_pdf(x, mu=mu, sigma=sigma) for mu in data]
    return np.mean(np.array(px), axis=0)


x = np.arange(-5, 12, 0.1)
prob = parzen_window_pdf(x, data, sigma=1)
plt.figure(figsize=(15,5))
plt.plot(x, prob)
plt.plot(data, [0]*len(data), '.r')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.grid()
plt.show()
plt.savefig('parzen_window_pdf.png')