'''
Created on Jun 1, 2011

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pca

dataArray = pca.loadDataSet('testSet.txt')
lowDArray, reconArray = pca.pca(dataArray, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataArray[:,0], dataArray[:,1], marker='^', s=90)
ax.scatter(reconArray[:,0], reconArray[:,1], marker='o', s=50, c='red')
plt.show()
