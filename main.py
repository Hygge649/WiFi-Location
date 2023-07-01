import pandas as pd
import numpy as np
np.random.seed(17)


path = "D:/desktop/c.c/UJIndoorLoc/trainingData.csv"

#Read the Flare dataset into a list and shuffle it with the random.shuffle method.
def readfile(path):
    data=pd.read_csv(path)
    data_array = np.array(data)  #数组
    np.random.shuffle(data_array)   #shuffle, fixed seed 按行重新洗牌
    column = np.arange(data_array.shape[1]) #  .shape[0])  raw
    np.random.shuffle(column)
    return data_array  #  training set:1795



dataSet1 = readfile(path)
dataSet=dataSet1[:1794, :200]
truth = dataSet1[:, 522]   #floor
# print(dataSet)
# print(dataSet.shape)
truth=set(truth)  #根据生成样本的类别来决定后面聚类的类别
k = len(truth)
print(truth)
print(k)


def randCent(dataSet,k):

    m,n = dataSet.shape #m=150,n=4
    centroids = np.zeros((k,n)) #3*4
    index1 = []
    for i in range(k): # 执行三次
        index = int(np.random.uniform(0,m)) # 产生0到150的随机数（在数据集中随机挑一个向量做为质心的初值）
        index1.append(index)
        centroids[i,:] = dataSet[index,:] #把对应行的十个维度传给质心的集合
    return centroids,index1

centroids,index = randCent(dataSet,k)
# print(centroids,index)

# 欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  # 计算欧氏距离

def cos(x,y):
    # 如果其中一个是零向量则直接返回
    if np.count_nonzero(x) == 0 or np.nonzero(y) == 0:
        return np.nan
    # 求其余弦距离
    d = np.dot(x,y) / ((np.sqrt(np.sum(x*x)) * np.sqrt(np.sum(y*y)))+float("1e-8"))
    #print((np.sqrt(np.sum(x*x)) * np.sqrt(np.sum(y*y)))+float("1e-8"))
    return d



# distance =cos(dataSet[0],dataSet[1])
# print(distance)


# k均值聚类算法
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]
    # 第一列存每个样本属于哪一簇
    # 第二列存每个样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))  # .mat()创建矩阵
    clusterChange = True
    # 1.初始化质心centroids
    centroids ,index= randCent(dataSet, k)  # 4*4
    while clusterChange:
        # 样本所属簇不再更新时停止迭代
        clusterChange = False
        # 遍历所有的样本
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有的质心
            # 2.找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离，找到距离最近的那个质心minIndex
                distance = cos(centroids[j], dataSet[i])
                #print(distance)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 3.更新该行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        #更新质心
        for j in range(k):
            # np.nonzero(x)返回值不为零的元素的下标，它的返回值是一个长度为x.ndim(x的轴数)的元组
            # 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。
            # 矩阵名.A 代表将 矩阵转化为array数组类型

            # 这里取矩阵clusterAssment所有行的第一列，转为一个array数组，与j（簇类标签值）比较，返回true or false
            # 通过np.nonzero产生一个array，其中是对应簇类所有的点的下标值（x个）
            # 再用这些下标值求出dataSet数据集中的对应行，保存为pointsInCluster（x*4）
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取对应簇类所有的点（x*4）
            if len(pointsInCluster):
                centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 求均值，产生新的质心
            # axis=0，那么输出是1行200列，求的是pointsInCluster每一列的平均值，即axis是几，那就表明哪一维度被压缩成1
        print("cluster complete")
        return centroids, clusterAssment


centroids, clusterAssment = KMeans(dataSet, k)

cluster = set()
for i in clusterAssment[:,0].tolist():
    for j in i:
        cluster.add(j)
print(cluster)  #聚类结果是三类

