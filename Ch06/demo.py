import svmMLiA
from numpy import *
dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
trace = True
bSimple, alphasSimple = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40, trace)
print '**************'
b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40, ('lin', 0), trace)
print bSimple, nonzero(alphasSimple)[0].shape
print b, nonzero(alphas)[0].shape
