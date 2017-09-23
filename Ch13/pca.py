'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return array(datArr)

def pca(dataArray, topNfeat=9999999):
    meanVals = mean(dataArray, axis=0)
    meanRemoved = dataArray - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(array(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = dot(meanRemoved, redEigVects)#transform data into new dimensions
    reconMat = dot(lowDDataMat, redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    dataArray = loadDataSet('secom.data', ' ')
    numFeat = shape(dataArray)[1]
    for i in range(numFeat):
        meanVal = mean(dataArray[nonzero(~isnan(dataArray[:,i]))[0],i]) #values that are not NaN (a number)
        dataArray[nonzero(isnan(dataArray[:,i]))[0],i] = meanVal  #set NaN values to mean
    return dataArray
