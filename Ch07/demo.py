from numpy import *
import adaboost

def testLoadDataSet(filename):
    dataArr, labelArr = adaboost.loadDataSet(filename)
    print array(dataArr).shape, array(labelArr).shape
    assert array(dataArr).shape == (67, 21)
    assert array(labelArr).shape == (67, )

def testBuildStump():
    D = ones(5)/5
    dataArr, classLabels = adaboost.loadSimpData()
    print adaboost.buildStump(dataArr, classLabels, D)

def testAdaBoost():
    dataArr, classLabels = adaboost.loadSimpData()
    classifierArr = adaboost.adaBoostTrainDS(dataArr, classLabels, 9)
    print classifierArr
    print adaboost.adaClassify([[0,0]],classifierArr)
    print adaboost.adaClassify([[5,5],[0,0]],classifierArr)
    print adaboost.adaClassify([[3,0]],classifierArr)

def tryTest(function):
    try:
        function()
    except RuntimeError, e:
        print 'catch RuntimeError[', e ,'] in function ', function.__name__

def testDraw():
    dataArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
    classifierArr, aggClassEst = adaboost.adaBoostTrainDS(dataArr, labelArr, 10, True)
    adaboost.plotROC(aggClassEst, labelArr)

if __name__ == "__main__":
    testLoadDataSet('horseColicTest2.txt')
    testBuildStump()
    testAdaBoost()
    tryTest(testDraw)
