import logRegres
from numpy import *

def functionForStudy():
    dataArr, labelArr = logRegres.loadDataSet()
    weights = logRegres.gradAscent(dataArr, labelArr, True)
    print weights
    labelArray = array(labelArr)
    sigmoidRes = logRegres.sigmoid(dot(dataArr,weights))
    error = sigmoidRes - labelArray
    # GOAL : choose weights so that the error is minimized.
    errorCount = (nonzero(abs(error) > 0.5)[0]).shape
    print errorCount[0]
    # but error is a vector, so how to define minimized?(Cost Function)
    # for example : using min errorCount; but how to adjust weights when using this goal?
    # actual choosed Cost function as following. using maximum likelihood estimate.
    costFunction = -vdot(labelArray, (log(sigmoidRes))) + vdot((1 - labelArray), log(array(1-sigmoidRes)))
    # J function is the average value of all costFunction values for all examples.
    J = sum(costFunction)
    print J

if __name__ == "__main__":
    functionForStudy()
