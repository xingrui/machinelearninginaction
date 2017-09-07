import apriori

def simpleTests():
    dataSet = apriori.loadDataSet()
    L, suppData = apriori.apriori(dataSet)
    for i in range(0, len(L)):
        print i, L[i]
    for k, v in suppData.iteritems():
        print k, v
    rules = apriori.generateRules(L, suppData, minConf=0.7)
    print rules

if __name__ == "__main__":
    simpleTests()
