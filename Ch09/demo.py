from numpy import *
import regTrees

def testSplitData():
    treeMat = eye(4) * 5
    treeMat = mat(eye(4) * 5)
    mat0, mat1 = regTrees.binSplitDataSet(treeMat, 0, 0.5)
    print mat0
    print mat1

def regressionTreeInner(filename, subplot, fig):
    import matplotlib
    import matplotlib.pyplot as plt
    myMat = mat(regTrees.loadDataSet(filename))
    ax = fig.add_subplot(subplot)
    ax.scatter(myMat.A[:,0], myMat.A[:,1],c='red')
    tree = regTrees.createTree(myMat)
    testDat = arange(min(myMat[:,0]),max(myMat[:,0]),0.01)
    yHat = regTrees.createForeCast(tree, testDat)
    ax.plot(testDat, yHat, linewidth = 2.0)
    print tree

def testModelInner(filename, subplot, fig):
    import matplotlib
    import matplotlib.pyplot as plt
    myMat2 = mat(regTrees.loadDataSet(filename))
    ax = fig.add_subplot(subplot)
    ax.scatter(myMat2.A[:,0], myMat2.A[:,1], c='red')
    myTree = regTrees.createTree(myMat2, regTrees.modelLeaf, regTrees.modelErr, (1,10))
    print myTree
    testDat = arange(min(myMat2[:,0]),max(myMat2[:,0]),0.01)
    yHat = regTrees.createForeCast(myTree, testDat, regTrees.modelTreeEval)
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
    myMat2 = mat(myDat2)
    myTree = regTrees.createTree(myMat2, ops=(0,1))
    myDat2Test = regTrees.loadDataSet('ex2test.txt')
    myMat2Test = mat(myDat2Test)
    print regTrees.prune(myTree, myMat2Test)

def testCompare():
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    trainMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
    testDat = regTrees.loadDataSet('bikeSpeedVsIq_test.txt')
    testDat.sort()
    testMat = mat(testDat)
    print testMat.shape
    myTree = regTrees.createTree(trainMat, ops=(1,20))
    yHat = regTrees.createForeCast(myTree, testMat[:,0])
    ax = fig.add_subplot(111)
    ax.scatter(trainMat.A[:,0], trainMat.A[:,1], c='red')
    ax.plot(testMat[:,0], yHat, linewidth = 2.0)
    print corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
    myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1,20))
    yHat = regTrees.createForeCast(myTree, testMat[:,0],regTrees.modelTreeEval)
    print corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
    ax.plot(testMat[:,0], yHat, linewidth = 2.0)
    plt.show()

if __name__ == "__main__":
    testRegression()
    testModel()
    testPrune()
    testCompare()
