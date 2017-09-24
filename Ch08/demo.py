import regression
from numpy import *

def normalRegression():
    xArr, yArr = regression.loadDataSet('ex0.txt')
    xArray = array(xArr)
    yArray = array(yArr)
    ws = regression.standRegres(xArray, yArray)
    yHat = dot(xArray,ws)
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

def abaloneTest():
    abX, abY = regression.loadDataSet('abalone.txt')
    paramList = [0.1, 1, 10]
    for param in paramList:
        yHat = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99],param)
        print '[lwlrRegress:%.1f] TrainError:' % param, regression.rssError(abY[0:99],yHat)
        yHat = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99],param)
        print '[lwlrRegress:%.1f] Test Error:' % param, regression.rssError(abY[100:199],yHat)
    ws = regression.standRegres(abX[0:99], abY[0:99])
    yHat = dot(abX[0:99], ws)
    print '[standRegress] Train Error:', regression.rssError(abY[0:99],yHat)
    yHat = dot(abX[100:199], ws) 
    print '[standRegress] Test Error:', regression.rssError(abY[100:199],yHat)

def ridgeTest():
    abX, abY = regression.loadDataSet('abalone.txt')
    ridgeWeights = regression.ridgeTest(abX, abY)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def stageWiseTest():
    xArr, yArr = regression.loadDataSet('abalone.txt')
    ridgeWeights = regression.stageWise(xArr, yArr, 0.005, 1000)
    xArray = array(xArr)
    yArray = array(yArr)
    xArray = regression.regularize(xArray)
    yArray -= mean(yArray)
    weights = regression.standRegres(xArray, yArray)
    print ridgeWeights[-1]
    print weights
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

#should only called once. (should not run again because lego.txt is in git now).
def initLegoDataSet():
    regression.setDataCollectFromHtml('lego.txt')

def crossValidationTest():
    lgX, lgY = regression.loadDataSet('lego.txt')
    m, n = array(lgX).shape
    lgX1 = array(ones((m,n+1)))
    lgX1[:,1:n+1] = array(lgX)
    ws = regression.standRegres(lgX1,lgY)
    print ws
    regression.crossValidation(lgX,lgY,10)
    regression.ridgeTest(lgX,lgY)

def tryTest(function):
    try:
        function()
    except RuntimeError, e:
        print 'catch RuntimeError[', e ,'] in function ', function.__name__

def main():
    tryTest(crossValidationTest)
    tryTest(stageWiseTest)
    tryTest(ridgeTest)
    tryTest(normalRegression)
    tryTest(lwlrRegression)
    abaloneTest()

if __name__ == "__main__":
    main()
