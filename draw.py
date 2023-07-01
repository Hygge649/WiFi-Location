# -*- coding: utf-8 -*-
from skkmeans import error_cknn
from loca import error_knn



import matplotlib.pyplot as plt


x = range(2,10)
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

y_1 = error_cknn
y_2 = error_knn # y轴的值


plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='cknn')
plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='knn')

plt.legend()  # 显示图例
plt.xlabel("Number of neighbor samples")  # X轴标签
plt.ylabel("Error")  # Y轴标签
plt.show()