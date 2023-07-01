import numpy as np
import pandas as pd
from sklearn.cluster import KMeans




#knn


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
data_train=dataSet[:1794, :200]



def cos(x,y):
    # 如果其中一个是零向量则直接返回
    if np.count_nonzero(x) == 0 or np.nonzero(y) == 0:
        return np.nan
    # 求其余弦距离
    d = np.dot(x,y) / ((np.sqrt(np.sum(x*x)) * np.sqrt(np.sum(y*y)))+float("1e-8"))
    #print((np.sqrt(np.sum(x*x)) * np.sqrt(np.sum(y*y)))+float("1e-8"))
    return d


import time
time_start = time.time()  # 记录开始时间
# function()   执行的程序
def test(data,k):
# test data[1]

    sim = []
    for i in range(len(data_train)):
        sim.append(cos(data, data_train[i]))

    sim = np.array(sim)
    sim = np.argsort(-sim)[0:k]  # 返回排序后的原数组的索引值，降序排列,选择10个最相似的
    #print('最相似的10个样本', sim)

    location_c = []
    for i in range(k):
        location_c.append(data_train[sim[i]])


    # 归一化，计算权重
    sim_sum = np.sum(sim)
    #print(sim_sum)


    w =[]
    for i in range(k):
        w.append(sim[i]/sim_sum)
    #print(len(w))

    #预测结果
    pre =0
    for i in range(k):
        pre += np.dot(w[i] , location_c[i])
    #print(pre[0],pre[1])

    # error
    l = (location[1][0] - pre[0]) / location[1][0]
    a = (location[1][1] - pre[1]) / location[1][1]
    return l,a


dataSet_test = readfile( "D:/desktop/c.c/UJIndoorLoc/validationData.csv")
location_test = dataSet_test[:179,520:522] #521 522 经度 纬度
data_test=dataSet_test[:179, :200]


#test(data_test[1],10)


#k_n
# x = []
# error_knn = []
# for j in range(2,10):
#     x.append(j)
#     l=0
#     a=0
#     for i in range(len(data_test)):
#         l= test(data_test[i],j)
#         l += abs(l)
#         #a += abs(a)
#     #print(l,a)
#
#     error_knn.append(l)
# #print(error)



#x,y
for i in range(len(data_test)):
    l,a= test(data_test[i],4)
    l += abs(l)
    a += abs(a)
print(l+a)



time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
#
# import matplotlib.pyplot as plt
#
#
# x = range(2,10)
# #plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
#
# y_1 = error_knn
# #y_2 = loca.error_knn # y轴的值
#
#
# plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='knn')
# #plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='knn')
#
# plt.legend()  # 显示图例
# plt.xlabel("Number of neighbor samples")  # X轴标签
# plt.ylabel("Error")  # Y轴标签
# plt.show()
#
#
#
