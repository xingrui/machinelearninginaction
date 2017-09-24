'''
Created on Mar 8, 2011

@author: Peter
'''
import operator
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = vdot(inA,inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataArray, user, simMeas, item):
    n = shape(dataArray)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataArray[user,j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataArray[:,item]>0, \
                                      dataArray[:,j]>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataArray[overLap,item], \
                                   dataArray[overLap,j])
        print 'standEst:the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
def svdEst(dataArray, user, simMeas, item):
    n = shape(dataArray)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataArray)
    Sig4 = diag(Sigma[:4])
    xformedItems = dot(dot(dataArray.T, U[:,:4]), la.inv(Sig4))  #create transformed items
    for j in range(n):
        userRating = dataArray[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item],\
                             xformedItems[j])
        print 'svdEst:the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recommend(dataArray, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataArray[user]==0)[0]#find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataArray, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=operator.itemgetter(1), reverse=True)[:N]

def printArray(inArray, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inArray[i,k]) > thresh:
                print 1,
            else: print 0,
        print ''

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        myl.append(map(int,line.strip()))
    myArray = array(myl)
    print "****original matrix******"
    printArray(myArray, thresh)
    U,Sigma,VT = la.svd(myArray)
    SigRecon = diag(Sigma[:numSV])
    reconArray = dot(dot(U[:,:numSV],SigRecon), VT[:numSV])
    print "****reconstructed matrix using %d singular values******" % numSV
    printArray(reconArray, thresh)
