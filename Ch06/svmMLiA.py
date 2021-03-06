'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep
import sys

def preprocess(dataArray, labelVector):
    storeDataArray = dot(dataArray, dataArray.T) # (M,N) dot (N,M) -> (M,M)
    storeLabelArray = outer(labelVector, labelVector)
    return multiply(storeDataArray, storeLabelArray)

# goal : maximize this calculateValue result
# under KKT conditions.
# 1 / sqrt(wTw) means the min value of distance from super plane and points(support vector points)
# it shows that the result is increasing, but the distance is not always increasing.
def calculateValue(alphas, storeArray):
    alphasArray = outer(alphas, alphas)
    wTw = vdot(alphasArray, storeArray)
    return sum(alphas) - 0.5 * wTw, 1 / sqrt(wTw)

# print value to stderr for differentiate from other logs.
def traceLog(trace, alphas, storeArray, indent=0):
    if not trace:
        return
    print >>sys.stderr, ' ' * indent, 'calculateValue:', calculateValue(alphas, storeArray)

def loadDataSet(fileName):
    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataArr.append([float(lineArr[0]), float(lineArr[1])])
        labelArr.append(float(lineArr[2]))
    return dataArr,labelArr

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataArr, classLabels, C, toler, maxIter, trace=False):
    dataArray = array(dataArr); labelVector = array(classLabels)
    b = 0; m,n = shape(dataArray)
    alphas = zeros(m)
    storeArray = preprocess(dataArray, labelVector)
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = vdot(multiply(alphas,labelVector), dot(dataArray,dataArray[i])) + b # (M,N) dot (N,) -> (M,)
            Ei = fXi - float(labelVector[i])#if checks if an example violates KKT conditions
            if ((labelVector[i]*Ei < -toler) and (alphas[i] < C)) or ((labelVector[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = vdot(multiply(alphas,labelVector), dot(dataArray,dataArray[j])) + b # (M,N) dot (N,) -> (M,)
                Ej = fXj - float(labelVector[j])
                alphaIold = alphas[i]; alphaJold = alphas[j];
                if (labelVector[i] != labelVector[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta = 2.0 * vdot(dataArray[i], dataArray[j]) - vdot(dataArray[i], dataArray[i]) - vdot(dataArray[j], dataArray[j])
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelVector[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelVector[j]*labelVector[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelVector[i]*(alphas[i]-alphaIold)*vdot(dataArray[i], dataArray[i]) - labelVector[j]*(alphas[j]-alphaJold)*vdot(dataArray[i], dataArray[j])
                b2 = b - Ej- labelVector[i]*(alphas[i]-alphaIold)*vdot(dataArray[i], dataArray[j]) - labelVector[j]*(alphas[j]-alphaJold)*vdot(dataArray[j], dataArray[j])
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                traceLog(trace, alphas, storeArray)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    traceLog(trace, alphas, storeArray, 4)
    return b,alphas

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    if kTup[0]=='lin': K = dot(X, A)   #linear kernel (M,N) dot (N,) -> (M,)
    elif kTup[0]=='rbf':
        delta = X - A #same as delta = X - tile(A, (shape(X)[0],1))
        K = square(delta).sum(axis=1)
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self, dataArray, labelVector, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataArray
        self.labelVector = labelVector
        self.C = C
        self.tol = toler
        self.m = shape(dataArray)[0]
        self.alphas = zeros(self.m)
        self.b = 0
        self.eCache = zeros((self.m,2)) #first column is valid flag
        self.K = zeros((self.m,self.m))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i], kTup)
        
def calcEk(oS, k):
    fXk = vdot(multiply(oS.alphas,oS.labelVector), oS.K[:,k]) + oS.b
    Ek = fXk - float(oS.labelVector[k])
    return Ek
        
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0])[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelVector[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelVector[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i]; alphaJold = oS.alphas[j];
        if (oS.labelVector[i] != oS.labelVector[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelVector[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelVector[j]*oS.labelVector[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelVector[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelVector[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelVector[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelVector[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataArr, classLabels, C, toler, maxIter,kTup=('lin', 0), trace=False):    #full Platt SMO
    oS = optStruct(array(dataArr),array(classLabels),C,toler, kTup)
    storeArray = preprocess(array(dataArr), array(classLabels))
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                innerRes = innerL(i,oS)
                alphaPairsChanged += innerRes
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                if innerRes : traceLog(trace, oS.alphas, storeArray)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas > 0) * (oS.alphas < C))[0]
            for i in nonBoundIs:
                innerRes = innerL(i,oS)
                alphaPairsChanged += innerRes
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                if innerRes : traceLog(trace, oS.alphas, storeArray)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    traceLog(trace, oS.alphas, storeArray, 4)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = array(dataArr); labelVector = array(classLabels)
    return dot(multiply(alphas, labelVector), X) # (M,) dot (M,N) -> (N,)

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    dataArray=array(dataArr); labelVector = array(labelArr)
    svInd=nonzero(alphas>0)[0]
    sVs=dataArray[svInd] #get matrix of only support vectors
    labelSV = labelVector[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(dataArray)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataArray[i],('rbf', k1))
        predict=vdot(kernelEval, multiply(labelSV,alphas[svInd])) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataArray=array(dataArr);
    m,n = shape(dataArray)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataArray[i],('rbf', k1))
        predict=vdot(kernelEval, multiply(labelSV,alphas[svInd])) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m)    
    
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for line in fr.readlines():
        returnVect.extend(map(int, line.strip()))
    return array(returnVect)

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    trainingArray = zeros((len(trainingFileList),1024))
    for i, fileNameStr in enumerate(trainingFileList):
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingArray[i] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingArray, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataArray=array(dataArr); labelVector=array(labelArr)
    svInd=nonzero(alphas>0)[0]
    sVs=dataArray[svInd] 
    labelSV = labelVector[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(dataArray)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataArray[i],kTup)
        predict=vdot(kernelEval, multiply(labelSV,alphas[svInd])) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    dataArray=array(dataArr);
    m,n = shape(dataArray)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataArray[i],kTup)
        predict=vdot(kernelEval, multiply(labelSV,alphas[svInd])) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m) 
