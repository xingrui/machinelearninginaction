from numpy import *
import kMeans 

def testbiKmeans():
    datMat = array(kMeans.loadDataSet('testSet2.txt'))
    centList, myNewAssments = kMeans.biKmeans(datMat, 3)
    print centList 

def testMap():
    kMeans.clusterClubs()

if __name__ == "__main__":
    testbiKmeans()
    testMap()
