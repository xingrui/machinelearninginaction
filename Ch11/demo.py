import apriori

def testAprioriGen():
    a=set('45')
    b=set('46')
    retList = apriori.aprioriGen([a,b],3)
    assert len(retList) == 1
    retList = apriori.aprioriGenWithBug([a,b],3)
    if len(retList) != 1:
        print 'should return 1 actual return', len(retList)

def simpleTests():
    dataSet = apriori.loadDataSet()
    L, suppData = apriori.apriori(dataSet)
    for i, item in enumerate(L):
        print i, item
    for k, v in suppData.iteritems():
        print k, v
    rules = apriori.generateRules(L, suppData, minConf=0.7)
    print rules

if __name__ == "__main__":
    simpleTests()
    testAprioriGen()
