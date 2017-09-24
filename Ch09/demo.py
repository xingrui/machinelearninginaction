from numpy import *
import regTrees

def testSplitData():
    treeArray = eye(4) * 5
    treeArray = array(eye(4) * 5)
    array0, array1 = regTrees.binSplitDataSet(treeArray, 0, 0.5)
    print array0 
    print array1

def regressionTreeInner(filename, subplot, fig):
    import matplotlib
    import matplotlib.pyplot as plt
    myArray = array(regTrees.loadDataSet(filename))
    ax = fig.add_subplot(subplot)
    ax.scatter(myArray[:,0], myArray[:,1],c='red')
    tree = regTrees.createTree(myArray)
    testDat = arange(min(myArray[:,0]),max(myArray[:,0]),0.01)
    yHat = regTrees.createForeCast(tree, testDat[:,newaxis])
    ax.plot(testDat, yHat, linewidth = 2.0)
    print tree

def testModelInner(filename, subplot, fig):
    import matplotlib
    import matplotlib.pyplot as plt
    myArray2 = array(regTrees.loadDataSet(filename))
    ax = fig.add_subplot(subplot)
    ax.scatter(myArray2[:,0], myArray2[:,1], c='red')
    myTree = regTrees.createTree(myArray2, regTrees.modelLeaf, regTrees.modelErr, (1,10))
    print myTree
    testDat = arange(min(myArray2[:,0]),max(myArray2[:,0]),0.01)
    yHat = regTrees.createForeCast(myTree, testDat[:,newaxis], regTrees.modelTreeEval)
    ax.plot(testDat, yHat, linewidth = 2.0)

def testFrame(InnerFunction):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    InnerFunction('ex00.txt', 221, fig)
    InnerFunction('ex0.txt', 223, fig)
    InnerFunction('sine.txt', 222, fig)
    InnerFunction('exp2.txt', 224, fig)
    plt.show()

def testRegression():
    testFrame(regressionTreeInner)

def testModel():
    testFrame(testModelInner)

def testPrune():
    myDat2 = regTrees.loadDataSet('ex2.txt')
    myArray2 = array(myDat2)
    myTree = regTrees.createTree(myArray2, ops=(0,1))
    myDat2Test = regTrees.loadDataSet('ex2test.txt')
    myArray2Test = array(myDat2Test)
    print regTrees.prune(myTree, myArray2Test)

def testCompare():
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    trainArray = array(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
    testDat = regTrees.loadDataSet('bikeSpeedVsIq_test.txt')
    testDat.sort()
    testArray = array(testDat)
    myTree = regTrees.createTree(trainArray, ops=(1,20))
    yHat = regTrees.createForeCast(myTree, testArray[:,0][:,newaxis])
    ax = fig.add_subplot(111)
    ax.scatter(trainArray[:,0], trainArray[:,1], c='red')
    ax.plot(testArray[:,0], yHat, linewidth = 2.0)
    print corrcoef(yHat, testArray[:,1], rowvar=0)[0,1]
    myTree = regTrees.createTree(trainArray, regTrees.modelLeaf, regTrees.modelErr, (1,20))
    yHat = regTrees.createForeCast(myTree, testArray[:,0][:,newaxis],regTrees.modelTreeEval)
    print corrcoef(yHat, testArray[:,1], rowvar=0)[0,1]
    ax.plot(testArray[:,0], yHat, linewidth = 2.0)
    plt.show()

def tryTest(function):
    try:
        function()
    except RuntimeError, e:
        print 'catch RuntimeError[', e ,'] in function ', function.__name__

if __name__ == "__main__":
    testSplitData()
    print 'testSplitData passed'
    tryTest(testRegression)
    print 'testRegression passed'
    tryTest(testModel)
    print 'testModel passed'
    testPrune()
    print 'testPrune passed'
    tryTest(testCompare)
    print 'testCompare passed'
