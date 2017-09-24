import kNN

def digitTests():
    kNN.handwritingClassTest()

def autoNormTests():
    datingDataArray,datingLabels = kNN.file2matrix('datingTestSet2.txt')
    normArray, ranges, minVals = kNN.autoNorm(datingDataArray)
    print normArray, ranges, minVals

def main():
    autoNormTests()
    digitTests()

if __name__ == "__main__":
    main()
