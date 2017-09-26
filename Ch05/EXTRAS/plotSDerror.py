'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logRegres

def stocGradAscent0(dataArray, classLabels):
    m,n = shape(dataArray)
    alpha = 0.5
    weightsVector = ones(n)   #initialize to all ones
    weightsHistory=zeros((500*m,n))
    for j in range(500):
        for i in range(m):
            h = logRegres.sigmoid(vdot(dataArray[i],weightsVector))
            error = classLabels[i] - h
            weightsVector += alpha * error * dataArray[i]
            weightsHistory[j*m + i] = weightsVector
    return weightsHistory

def stocGradAscent1(dataArray, classLabels):
    m,n = shape(dataArray)
    alpha = 0.4
    weightsVector = ones(n)   #initialize to all ones
    weightsHistory=zeros((40*m,n))
    for j in range(40):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = logRegres.sigmoid(vdot(dataArray[randIndex],weightsVector))
            error = classLabels[randIndex] - h
            #print error
            weightsVector += alpha * error * dataArray[randIndex]
            weightsHistory[j*m + i] = weightsVector
            del(dataIndex[randIndex])
    print weightsVector
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
