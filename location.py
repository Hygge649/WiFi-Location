import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

def cluster(data,k):
    # 聚类数量

    # 训练模型
    model = KMeans(n_clusters=k)
    model.fit(data)
    # 预测结果
    result = model.predict(data)  #返回类别
    #print(len(result))
    #查看质心
    centroid=model.cluster_centers_
    #print(len(centroid))
    inertia=model.inertia_#J
    # print(inertia)  #样本到质心的距离之和
    return inertia




import matplotlib.pyplot as plt



#plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
x = []
y_1 = []
for i in range(2,20):
    x.append(i)
    y_1.append(cluster(data,i))
#y_2 = loca.error_knn # y轴的值


plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='J')
#plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='knn')

plt.legend()  # 显示图例
plt.xlabel("Number of clustering")  # X轴标签
plt.ylabel("J")  # Y轴标签
plt.show()
