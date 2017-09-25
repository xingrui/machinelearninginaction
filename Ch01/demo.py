from numpy import *

# When operating on two arrays, NumPy compares their shapes element-wise.
# It starts with the trailing dimensions, and works its way forward.
# Two dimensions are compatible when
# 1.they are equal, or
# 2.one of them is 1
def testBroadcast():
    dataArray = array(([1,2,3],[6,7,8]))
    minusResult = dataArray - array([2,0,0])
    minusResultTile = dataArray - tile(array([2,0,0]), (2,1))
    assert (minusResult == minusResultTile).all()
    a = array([1,2,3,4])
    b = array([1,2,3])
    l = a[:,newaxis]
    r = b[newaxis,:]
    print l.shape, r.shape
    assert l.shape == (4,1)
    assert r.shape == (1,3)
    assert (l+r).shape == (4,3)
    assert (l-r).shape == (4,3)
    assert multiply(l,r).shape == (4,3)
    assert (l/r).shape == (4,3)

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
    testMat = matrix(testArray)
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
    testMat = matrix(testArray)
    nonzeroIndex = nonzero(testMat[:,1]>150)
    assert len(nonzeroIndex) == 2
    subIndex = nonzeroIndex[0]
    assert (subIndex == matrix([[1,3]])).all()
    uglyMat = testMat[subIndex]
    assert uglyMat.shape == (1,2,2) # WHAT THE FUCK !!!!!!
    assert uglyMat[0].shape == (2,2)

# outer and dot function
def testUsefulFunctions():
    b = array([1,2,3])
    print outer(b, b)
    # outer function is the simplest way to show outer product of two vector.
    assert (outer(b,b) == multiply(b[:,newaxis],b[newaxis,:])).all()
    assert (outer(b,b) == dot(b[:,newaxis],b[newaxis,:])).all()
    assert (outer(b,b) == multiply(reshape(b,(-1,1)),reshape(b,(1,-1)))).all()
    assert (outer(b,b) == dot(reshape(b,(-1,1)),reshape(b,(1,-1)))).all()
    # the 1-d array will changed as use in dot function.
    array_dot = dot(array([[1,2,3],[4,5,7]]), array([1,2,3]))
    assert array_dot.shape == (2,)
    assert (array_dot == array([14,35])).all()
    array_dot = dot(array([1,2]),array([[1,2,3],[4,5,7]]))
    assert array_dot.shape == (3,)
    assert (array_dot == array([9,12,17])).all()

def innerTest(dataArray, t):
    dataArray = dataArray.astype(t)
    try:
        res1 = vdot(dataArray, dataArray)
        res2 = vdot(dataArray[0], dataArray[1])
        assert res1 == sum(multiply(dataArray, dataArray))
        assert res2 == sum(multiply(dataArray[0], dataArray[1]))
        return 0
    except ValueError,e:
        return 1

def testVdotArray():
    dataArray = 2 * ones((5,5))
    type_list = ['int', 'int8', 'int16', 'int32', 'int64', 'float', 'float16', 'float32','float64','float128', 'complex', 'complex64', 'complex128', 'complex256']
    res_list = [[],[]]
    for t in type_list:
        ret = innerTest(dataArray, t)
        res_list[ret].append(t)
    assert len(res_list[1]) == 0, '!!!!!! vdot of array will have peoblem except_list: %s' % str(res_list[1])

# vdot of matrix may have problem !!!!!!
def testVdotMatrix():
    dataArray = 2 * ones((5,5))
    dataMat = matrix(dataArray)
    type_list = ['int', 'int8', 'int16', 'int32', 'int64', 'float', 'float16', 'float32','float64','float128', 'complex', 'complex64', 'complex128', 'complex256']
    res_list = [[],[]]
    for t in type_list:
        ret = innerTest(dataMat, t)
        res_list[ret].append(t)
    print 'passed_list:', res_list[0]
    assert len(res_list[1]) == 0, '!!!!!! vdot of matrix will have peoblem except_list: %s' % str(res_list[1])

def tryTest(function):
    print function.__name__, 'begin.'
    try:
        function()
        print function.__name__, 'passed.'
    except AssertionError, e:
        print 'CATCH EXCEPTION:', e
        exit(0)

def main():
    tryTest(testBroadcast)
    tryTest(testSquare)
    tryTest(testStar)
    tryTest(testShape)
    tryTest(testSetOperation)
    tryTest(testRemoveCertainRow)
    tryTest(testUsefulFunctions)
    tryTest(testVdotArray)
    tryTest(testVdotMatrix)

if __name__ == "__main__":
    main()
