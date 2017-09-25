from numpy import *

def testSquare():
    print 'testSquare begin.'
    testArray = array([[1,1],[1,1]])
    testMat = mat(testArray)
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
    print 'testSquare passed.'

# star(*) will be have ambiguity.(different meaning when using different data type)
# so using dot or multiply function in code will be much more readable.
def testStar():
    print 'testStar begin.'
    testArray = array([[4,3], [2,1]])
    testMat = mat(testArray)
    dotRes = array([[22,15], [10,7]])
    multiplyRes = array([[16,9], [4,1]])
    assert (dot(testArray, testArray) == dotRes).all()
    assert (dot(testMat, testMat) == dotRes).all()
    assert (multiply(testArray, testArray) == multiplyRes).all()
    assert (multiply(testMat, testMat) == multiplyRes).all()
    assert (testArray * testArray == multiplyRes).all()
    assert (testMat * testMat == dotRes).all()
    print 'testStar passed.'

def main():
    testSquare()
    testStar()

if __name__ == "__main__":
    main()
