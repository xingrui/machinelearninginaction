from numpy import *
import adaboost

def testLoadDataSet(filename):
    dataMat, labelMat = adaboost.loadDataSet(filename)
    print mat(dataMat).shape, mat(labelMat).shape
    assert mat(dataMat).shape == (67, 21)
    assert mat(labelMat).shape == (1, 67)

def testBuildStump():
    D = mat(ones((5,1))/5)
    print D
    datMat, classLabels = adaboost.loadSimpData()
    print adaboost.buildStump(datMat, classLabels, D)

if __name__ == "__main__":
    testLoadDataSet('horseColicTest2.txt')
    testBuildStump()
