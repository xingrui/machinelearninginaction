import kNN

def digitTests():
    kNN.handwritingClassTest()

def autoNormTests():
    datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    print normMat, ranges, minVals

def main():
    autoNormTests()
    digitTests()

if __name__ == "__main__":
    main()
