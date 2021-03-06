'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from collections import Counter
from os import listdir

def classify0(inX, dataSet, labels, k):
    diffArray = dataSet - inX # same as diffArray = dataSet - tile(inX, (dataSet.shape[0],1))
    sqDistances = square(diffArray).sum(axis=1)
    distances = sqrt(sqDistances)
    sortedDistIndicies = distances.argsort()     
    voteIlabelList = [labels[x] for x in sortedDistIndicies[:k]]
    return Counter(voteIlabelList).most_common(1)[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnArray = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelArr = []                       #prepare labels return   
    index = 0
    for line in arrayOLines:
        listFromLine = line.strip().split('\t')
        returnArray[index] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelArr.append(int(listFromLine[-1]))
        else:
            classLabelArr.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnArray,classLabelArr

    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = (dataSet - minVals) / ranges # skip tile operation.
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataArray,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normArray, ranges, minVals = autoNorm(datingDataArray)
    m = normArray.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normArray[i],normArray[numTestVecs:m],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataArray, datingLabels = file2matrix('datingTestSet2.txt')
    normArray, ranges, minVals = autoNorm(datingDataArray)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normArray, datingLabels, 3)
    print "You will probably like this person: %s" % resultList[classifierResult - 1]
    
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for line in fr.readlines():
        returnVect.extend(map(int, line.strip()))
    return array(returnVect)

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    trainingArray = zeros((len(trainingFileList),1024))
    for i, fileNameStr in enumerate(trainingFileList):
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingArray[i] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    for i, fileNameStr in enumerate(testFileList):
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingArray, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(len(testFileList)))
