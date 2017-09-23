from numpy import *
import adaboost

def testLoadDataSet(filename):
    dataMat, labelMat = adaboost.loadDataSet(filename)
    print array(dataMat).shape, array(labelMat).shape
    assert array(dataMat).shape == (67, 21)
    assert array(labelMat).shape == (67, )

def testBuildStump():
    D = ones(5)/5
    datMat, classLabels = adaboost.loadSimpData()
    print adaboost.buildStump(datMat, classLabels, D)

def testAdaBoost():
    datMat, classLabels = adaboost.loadSimpData()
    classifierArray = adaboost.adaBoostTrainDS(datMat, classLabels, 9)
    print classifierArray
    print adaboost.adaClassify([[0,0]],classifierArray)
    print adaboost.adaClassify([[5,5],[0,0]],classifierArray)
    print adaboost.adaClassify([[3,0]],classifierArray)

if __name__ == "__main__":
    testLoadDataSet('horseColicTest2.txt')
    testBuildStump()
    testAdaBoost()
