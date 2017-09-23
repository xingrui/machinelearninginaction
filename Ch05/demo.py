import logRegres
dataArr, labelArr = logRegres.loadDataSet()
logRegres.gradAscent(dataArr, labelArr, True)
logRegres.colicTest()
#logRegres.multiTest()
