import regression
from numpy import *

def normalRegression():
    xArr, yArr = regression.loadDataSet('ex0.txt')
    xArray = array(xArr)
    yArray = array(yArr)
    ws = regression.standRegres(xArray, yArray)
    yHat = dot(xArray,ws).T.A[0]
    print corrcoef(yHat, yArray)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xArray[:,1],yArray)
    ax.plot(xArray[:,1],yHat)
    plt.show()

def lwlrRegression():
    xArr, yArr = regression.loadDataSet('ex0.txt')
    xArray = array(xArr)
    yArray = array(yArr)
    srtInd = xArray[:,1].argsort(0)
    xSort = xArray[srtInd]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    param = [1.0, 0.04, 0.01, 0.003]
    for i in xrange(len(param)):
        yHat = regression.lwlrTest(xArray,xArray, yArray,param[i])
        ax = fig.add_subplot(2,2,i+1)
        ax.scatter(xArray[:,1],yArray,s=2,c='red')
        ax.plot(xSort[:,1],yHat[srtInd])
    plt.show()

def main():
    normalRegression()
    lwlrRegression()

if __name__ == "__main__":
    main()
