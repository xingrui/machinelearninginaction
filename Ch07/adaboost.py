'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    dataArray = array([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataArray,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataArr.append(map(float, curLine[:-1]))
        labelArr.append(float(curLine[-1]))
    return dataArr,labelArr

def stumpClassify(dataArray,dimen,threshVal,threshIneq):#just classify the data
    retVector = ones(shape(dataArray)[0])
    if threshIneq == 'lt':
        retVector[dataArray[:,dimen] <= threshVal] = -1.0
    else:
        retVector[dataArray[:,dimen] > threshVal] = -1.0
    return retVector
    

def buildStump(dataArr,classLabels,D):
    dataArray = array(dataArr); labelVector = array(classLabels)
    m,n = shape(dataArray)
    numSteps = 10.0; bestStump = {}; bestClasEst = zeros(m)
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataArray[:,i].min(); rangeMax = dataArray[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataArray,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = ones(m)
                errArr[predictedVals == labelVector] = 0
                weightedError = vdot(D,errArr)  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40,retAgg=False):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = ones(m)/m   #init D to all equal
    aggClassEst = zeros(m)
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst
        expon = multiply(-1*alpha*array(classLabels),classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst
        aggErrors = multiply(sign(aggClassEst) != array(classLabels),ones(m))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: break
    if retAgg:return weakClassArr, aggClassEst
    else:return weakClassArr

def adaClassify(datToClass,classifierArr):
    dataArray = array(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataArray)[0]
    aggClassEst = zeros(m)
    for classifier in classifierArr:
        classEst = stumpClassify(dataArray, classifier['dim'],\
                                 classifier['thresh'],\
                                 classifier['ineq'])#call stump classify
        aggClassEst += classifier['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep
