'''
Created on Jun 1, 2011

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pca

dataMat = pca.loadDataSet('testSet.txt')
lowDMat, reconMat = pca.pca(dataMat, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat.A[:,0], dataMat.A[:,1], marker='^', s=90)
ax.scatter(reconMat.A[:,0], reconMat.A[:,1], marker='o', s=50, c='red')
plt.show()
