'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataArr.append(map(float, curLine[:-1]))
        labelArr.append(float(curLine[-1]))
    return dataArr,labelArr

def standRegres(xArr,yArr):
    xArray = array(xArr); yArray = array(yArr)
    xTx = dot(xArray.T, xArray)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = dot(linalg.inv(xTx), dot(xArray.T, yArray))
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xArray = array(xArr); yArray = array(yArr)
    m = shape(xArray)[0]
    weights = eye(m)
    for j in range(m):                      #next 2 lines create weights matrix
        diffArray = testPoint - xArray[j]     #
        weights[j,j] = exp(sum(square(diffArray))/(-2.0*k**2))
    xTx = dot(xArray.T, dot(weights, xArray))
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = dot(linalg.inv(xTx), dot(xArray.T, dot(weights, yArray)))
    return dot(testPoint, ws)

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    testArray = array(testArr)
    m = shape(testArray)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArray[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = array(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return sum(square(yArr-yHatArr))

def ridgeRegres(xArray,yArray,lam=0.2):
    xTx = dot(xArray.T, xArray)
    denom = xTx + eye(shape(xArray)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = dot(linalg.inv(denom), dot(xArray.T, yArray))
    return ws
    
def ridgeTest(xArr,yArr):
    xArray = array(xArr); yArray=array(yArr)
    yArray -= mean(yArray)
    xArray = regularize(xArray)
    numTestPts = 30
    wArray = zeros((numTestPts,shape(xArray)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xArray,yArray,exp(i-10))
        wArray[i]=ws
    return wArray

def regularize(xArray):#regularize by columns
    inMeans = mean(xArray,0)   #calc mean then subtract it off
    inVar = var(xArray,0)      #calc variance of Xi then divide by it
    return (xArray - inMeans)/inVar

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xArray = array(xArr); yArray=array(yArr)
    yArray -= mean(yArray)
    xArray = regularize(xArray)
    m,n=shape(xArray)
    returnArray = zeros((numIt,n)) #testing code remove
    ws = zeros(n); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        #print ws
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = dot(xArray, wsTest)
                rssE = rssError(yArray,yTest)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnArray[i]=ws
    return returnArray

def scrapePage(inFile,outFile,yr,numPce,origPrc):
    from bs4 import BeautifulSoup
    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
    soup = BeautifulSoup(fr.read())
    i=1
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print "item #%d did not sell" % i
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
            print "%s\t%d\t%s" % (priceStr,newFlag,title)
            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)
    fw.close()
    
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    import urllib2
    import json
    from time import sleep
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for currItem in retDict['items']:
        try:
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i

# should only called once. the outfile is stored into lego.txt in git file.
def setDataCollectFromHtml(outFileName):
    scrapePage('setHtml/lego8288.html',outFileName, 2006, 800, 49.99)
    scrapePage('setHtml/lego10030.html',outFileName, 2002, 3096, 269.99)
    scrapePage('setHtml/lego10179.html',outFileName, 2007, 5195, 499.99)
    scrapePage('setHtml/lego10181.html',outFileName, 2007, 3428, 199.99)
    scrapePage('setHtml/lego10189.html',outFileName, 2008, 5922, 299.99)
    scrapePage('setHtml/lego10196.html',outFileName, 2009, 3263, 249.99)
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
    
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorArray = zeros((numVal,30))#create error matrix 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wArray= ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            arrayTestX = array(testX); arrayTrainX=array(trainX)
            meanTrain = mean(arrayTrainX,0)
            varTrain = var(arrayTrainX,0)
            arrayTestX = (arrayTestX-meanTrain)/varTrain #regularize test with training params
            yEst = dot(arrayTestX, wArray[k]) + mean(trainY)#test ridge results and store
            errorArray[i,k]=rssError(yEst,array(testY))
            #print errorArray[i,k]
    meanErrors = mean(errorArray,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wArray[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  dot(x,w)/var(x) - meanX/var(x) +meanY
    xArray = array(xArr); yArray=array(yArr)
    meanX = mean(xArray,0); varX = var(xArray,0)
    unReg = bestWeights/varX
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*vdot(meanX,unReg) + mean(yArray)
