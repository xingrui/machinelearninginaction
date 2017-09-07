import fpGrowth

def tests():
    simpDat = fpGrowth.loadSimpDat()
    initSet = fpGrowth.createInitSet(simpDat)
    myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)
    myFPtree.disp()
    freqItems = []
    fpGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print freqItems

if __name__ == "__main__":
    tests()
