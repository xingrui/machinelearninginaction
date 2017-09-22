from numpy import *
import adaboost

def testDraw():
    datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(datArr, labelArr, 10, True)
    adaboost.plotROC(aggClassEst, labelArr)

if __name__ == "__main__":
    testDraw()
