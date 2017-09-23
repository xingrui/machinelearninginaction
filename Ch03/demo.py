import trees
from numpy import *

def tests():
    dataSet, labels = trees.createDataSet()
    print dataSet
    print trees.calcShannonEnt(dataSet)
    t = trees.createTree(dataSet, labels)
    print t

def copyTest():
    # list test.
    a = [1,2,3,4]
    b = a[:]
    a[3] = 5
    assert b[3] == 4
    #array test.
    a = array([1,2,3,4])
    b = a[:]
    a[3] = 5
    assert b[3] == 5
    print 'assert passed.'

def main():
    copyTest()
    tests()

if __name__ == "__main__":
    main()
