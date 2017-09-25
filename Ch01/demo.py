from numpy import *

def testSquare():
    testArray = array([[1,1],[1,1]])
    testMat = matrix(testArray)
    # ** 2 will not always use element square.
    assert (testArray ** 2 == testArray).all()
    assert (testMat ** 2 != testMat).all()
    assert (testMat ** 2 == 2 * testMat).all()
    # square will always use element square.
    assert (square(testArray) == testArray).all()
    assert (square(testMat) == testMat).all()
    # power 2 will always use element square.
    assert (power(testArray,2) == testArray).all()
    assert (power(testMat,2) == testMat).all()

# star(*) will be have ambiguity.(different meaning when using different data type)
# so using dot or multiply function in code will be much more readable.
def testStar():
    testArray = array([[4,3], [2,1]])
    testMat = matrix(testArray)
    dotRes = array([[22,15], [10,7]])
    multiplyRes = array([[16,9], [4,1]])
    assert (dot(testArray, testArray) == dotRes).all()
    assert (dot(testMat, testMat) == dotRes).all()
    assert (multiply(testArray, testArray) == multiplyRes).all()
    assert (multiply(testMat, testMat) == multiplyRes).all()
    assert (testArray * testArray == multiplyRes).all()
    assert (testMat * testMat == dotRes).all()

# matrix [:,0] will return 2-d matrix.
# array [:,0] will return 1-d array.
def testShape():
    testArray = array([[1,2,3],[2,5,6]])
    testMat = matrix(testArray)
    #array return 1-d array.
    assert testArray[0].shape == (3,)
    assert testArray[0].T.shape == (3,)
    assert testArray[:,0].shape == (2,)
    assert testArray[:,0].T.shape == (2,)
    #matrix return 2-d matrix.
    assert testMat[0].shape == (1,3)
    assert testMat[0].T.shape == (3,1)
    assert testMat[:,0].shape == (2,1)
    assert testMat[:,0].T.shape == (1,2)
    #test minus mean value.
    minusAxis0Array = testArray - testArray.mean(axis=0)
    minusAxis1Array = testArray - testArray.mean(axis=1)[:,newaxis] # need [:,newaxis]
    minusAxis0Mat = testMat - testMat.mean(axis=0)
    minusAxis1Mat = testMat - testMat.mean(axis=1)
    assert (minusAxis0Array == minusAxis0Mat).all()
    assert (minusAxis1Array == minusAxis1Mat).all()

def testSetOperation():
    testArray = array([100,200,100,200])
    testMat = mat(testArray)
    print set(testArray)
    assert len(set(testArray)) == 2
    print set(testMat.T)
    assert len(set(testMat.T)) == 4 # not 2 !!!!!!

def testRemoveCertainRow():
    # following code is clear.
    testArray = array([[1,100],[1,200],[2,100],[2,200]])
    nonzeroIndex = nonzero(testArray[:,1]>150)
    assert len(nonzeroIndex) == 1
    subIndex = nonzeroIndex[0]
    assert (subIndex == array([1,3])).all()
    assert (testArray[subIndex] == array([[1,200],[2,200]])).all()
    # following code is very ugly and confusing !!!!!!
    # matrix is strongly not recommended when using numpy !!!!!!
    testMat = mat(testArray)
    nonzeroIndex = nonzero(testMat[:,1]>150)
    assert len(nonzeroIndex) == 2
    subIndex = nonzeroIndex[0]
    assert (subIndex == matrix([[1,3]])).all()
    uglyMat = testMat[subIndex]
    assert uglyMat.shape == (1,2,2) # WHAT THE FUCK !!!!!!
    assert uglyMat[0].shape == (2,2)

def tryTest(function):
    print function.__name__, 'begin.'
    try:
        function()
        print function.__name__, 'passed.'
    except AssertionError, e:
        print 'CATCH EXCEPTION:', e

def main():
    tryTest(testSquare)
    tryTest(testStar)
    tryTest(testShape)
    tryTest(testSetOperation)
    tryTest(testRemoveCertainRow)

if __name__ == "__main__":
    main()
