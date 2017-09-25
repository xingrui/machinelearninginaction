'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

def loadDataSet():
    dataArr = []; labelArr = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataArr.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelArr.append(int(lineArr[2]))
    return dataArr,labelArr

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''
logitisc regression goal :
minimized the total/average cost for following calculateCost function.
'''
def calculateCost(labelVector, sigmoidVector):
    costFunction = -(vdot(labelVector, log(sigmoidVector)) + vdot((1 - labelVector), log(1 - sigmoidVector)))
    return costFunction

def gradAscent(dataArrIn, classLabels, trace=False):
    dataArray = array(dataArrIn)
    labelVector = array(classLabels) 
    m,n = shape(dataArray)
    alpha = 0.001
    maxCycles = 500
    weightsVector = ones(n)
    for k in range(maxCycles):              #heavy on matrix operations
        hVector = sigmoid(dot(dataArray,weightsVector))     #matrix mult (M,N) dot (N,) -> (M,)
        errorVector = (labelVector - hVector)              #vector subtraction
        weightsVector = weightsVector + alpha * dot(dataArray.T, errorVector) #matrix mult (N,M) dot (M,) -> (N,)
        # print total cost for each step. and we will see that the cost is decreasing.
        if trace:print calculateCost(labelVector, hVector)
    return weightsVector

def plotBestFit(weightsVector):
    import matplotlib.pyplot as plt
    dataArr,labelArr=loadDataSet()
    dataArr = array(dataArr)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelArr[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weightsVector[0]-weightsVector[1]*x)/weightsVector[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataArray, classLabels):
    m,n = shape(dataArray)
    alpha = 0.01
    weightsVector = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(vdot(dataArray[i],weightsVector))
        error = classLabels[i] - h
        weightsVector = weightsVector + alpha * error * dataArray[i]
    return weightsVector

def stocGradAscent1(dataArray, classLabels, numIter=150):
    m,n = shape(dataArray)
    weightsVector = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(vdot(dataArray[randIndex],weightsVector))
            error = classLabels[randIndex] - h
            weightsVector = weightsVector + alpha * error * dataArray[randIndex]
            del(dataIndex[randIndex])
    return weightsVector

def classifyVector(inX, weightsVector):
    prob = sigmoid(vdot(inX,weightsVector))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        trainingSet.append(map(float, currLine[:-1]))
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = map(float, currLine[:-1])
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
        
