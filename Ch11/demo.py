import apriori

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
