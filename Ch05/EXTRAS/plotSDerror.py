'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logRegres

def stocGradAscent0(dataArrayay, classLabels):
    m,n = shape(dataArrayay)
    alpha = 0.5
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((500*m,n))
    for j in range(500):
        for i in range(m):
            h = logRegres.sigmoid(sum(dataArrayay[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataArrayay[i]
            weightsHistory[j*m + i] = weights
    return weightsHistory

def stocGradAscent1(dataArrayay, classLabels):
    m,n = shape(dataArrayay)
    alpha = 0.4
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((40*m,n))
    for j in range(40):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = logRegres.sigmoid(sum(dataArrayay[randIndex]*weights))
            error = classLabels[randIndex] - h
            #print error
            weights = weights + alpha * error * dataArrayay[randIndex]
            weightsHistory[j*m + i] = weights
            del(dataIndex[randIndex])
    print weights
    return weightsHistory
    

dataArr,labelArr=logRegres.loadDataSet()
dataArray = array(dataArr)
myHist = stocGradAscent1(dataArray,labelArr)


n = shape(dataArray)[0] #number of points to create
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []

markers =[]
colors =[]


fig = plt.figure()
ax = fig.add_subplot(311)
type1 = ax.plot(myHist[:,0])
plt.ylabel('X0')
ax = fig.add_subplot(312)
type1 = ax.plot(myHist[:,1])
plt.ylabel('X1')
ax = fig.add_subplot(313)
type1 = ax.plot(myHist[:,2])
plt.xlabel('iteration')
plt.ylabel('X2')
plt.show()
