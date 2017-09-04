import logRegres
from numpy import *

def functionForStudy():
    dataArr, labelMat = logRegres.loadDataSet()
    weights = logRegres.gradAscent(dataArr, labelMat, True)
    print weights.A
    labelH = mat(labelMat).T
    labelArr = array(labelH)
    sigmoidRes = logRegres.sigmoid(dataArr*weights)
    error = sigmoidRes - labelH
    # GOAL : choose weights so that the error is minimized.
    errorCount = (nonzero(abs(error) > 0.5)[0]).shape
    print errorCount[1]
    # but error is a vector, so how to define minimized?(Cost Function)
    # for example : using min errorCount; but how to adjust weights when using this goal?
    # actual choosed Cost function as following. using maximum likelihood estimate.
    costFunction = -(labelArr * (log(array(sigmoidRes))) + (1 - labelArr) * log(array(1-sigmoidRes)))
    # J function is the average value of all costFunction values for all examples.
    J = sum(costFunction)
    print J

if __name__ == "__main__":
    functionForStudy()
