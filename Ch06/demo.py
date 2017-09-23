import svmMLiA
from numpy import *

def simpleExample():
    dataArr = array(([-10,0],[0,0],[6,8],[20,0]))
    labelArr = [1, 1, -1, -1]
    trace = True
    bSimple, alphasSimple = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40, trace)
    print '**************'
    b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40, ('lin', 0), trace)
    print bSimple, nonzero(alphasSimple)[0].shape
    print b, nonzero(alphas)[0].shape
    ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
    print ws
    print dot(dataArr[0], ws) + b

def detailTest():
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    trace = True
    bSimple, alphasSimple = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40, trace)
    print '**************'
    b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40, ('lin', 0), trace)
    print bSimple, nonzero(alphasSimple)[0].shape
    print b, nonzero(alphas)[0].shape

def tests():
    svmMLiA.testRbf()
    svmMLiA.testDigits(('rbf', 20))

if __name__ == "__main__":
    detailTest()
    simpleExample()
    tests()
