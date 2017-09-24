from numpy import *
import kMeans 

def testbiKmeans():
    datMat = array(kMeans.loadDataSet('testSet2.txt'))
    centList, myNewAssments = kMeans.biKmeans(datMat, 3)
    print centList 

def testMap():
    kMeans.clusterClubs()

def tryTest(function):
    try:
        function()
    except RuntimeError, e:
        print 'catch RuntimeError[', e ,'] in function ', function.__name__

if __name__ == "__main__":
    testbiKmeans()
    tryTest(testMap)
