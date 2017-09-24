'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataArr = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataArr.append(fltLine)
    return dataArr

def binSplitDataSet(dataArray, feature, value):
    array0 = dataArray[nonzero(dataArray[:,feature] > value)[0]]
    array1 = dataArray[nonzero(dataArray[:,feature] <= value)[0]]
    return array0,array1

def regLeaf(dataArray):#returns the value used for each leaf
    return mean(dataArray[:,-1])

def regErr(dataArray):
    return var(dataArray[:,-1]) * shape(dataArray)[0]

def linearSolve(dataArray):   #helper function used in two places
    m,n = shape(dataArray)
    X = ones((m,n)); 
    X[:,1:n] = dataArray[:,0:n-1]; Y = dataArray[:,-1]#and strip out Y
    xTx = dot(X.T, X)
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = dot(linalg.inv(xTx), dot(X.T, Y))
    return ws,X,Y

def modelLeaf(dataArray):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataArray)
    return ws

def modelErr(dataArray):
    ws,X,Y = linearSolve(dataArray)
    yHat = dot(X, ws)
    return sum(square(Y - yHat))

def chooseBestSplit(dataArray, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataArray[:,-1])) == 1: #exit cond 1
        return None, leafType(dataArray)
    m,n = shape(dataArray)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataArray)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataArray[:,featIndex]):
            array0, array1 = binSplitDataSet(dataArray, featIndex, splitVal)
            if (shape(array0)[0] < tolN) or (shape(array1)[0] < tolN): continue
            newS = errType(array0) + errType(array1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataArray) #exit cond 2
    array0, array1 = binSplitDataSet(dataArray, bestIndex, bestValue)
    if (shape(array0)[0] < tolN) or (shape(array1)[0] < tolN):  #exit cond 3
        return None, leafType(dataArray)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataArray is NumPy array so we can array filtering
    dataArray = array(dataSet)
    feat, val = chooseBestSplit(dataArray, leafType, errType, ops)#choose the best split

    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataArray, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):
    return (type(obj) == dict)

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(square(lSet[:,-1] - tree['left'])) +\
            sum(square(rSet[:,-1] - tree['right']))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(square(testData[:,-1] - treeMean))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):
    return model

def modelTreeEval(model, inDat):
    n = shape(inDat)[0]
    X = ones(n+1)
    X[1:n+1]=inDat
    return vdot(X,model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = testData.shape[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = treeForeCast(tree, testData[i], modelEval)
    return yHat
