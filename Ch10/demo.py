from numpy import *
import kMeans 

def testbiKmeans():
    datMat = mat(kMeans.loadDataSet('testSet2.txt'))
    centList, myNewAssments = kMeans.biKmeans(datMat, 3)
    print centList 

if __name__ == "__main__":
    testbiKmeans()
