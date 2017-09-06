from numpy import *
import adaboost

def testLoadDataSet(filename):
    dataMat, labelMat = adaboost.loadDataSet(filename)
    print mat(dataMat).shape, mat(labelMat).shape
    print dataMat
    assert mat(dataMat).shape == (67, 21)
    assert mat(labelMat).shape == (1, 67)

if __name__ == "__main__":
    testLoadDataSet('horseColicTest2.txt')
