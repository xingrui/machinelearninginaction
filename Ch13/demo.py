import pca
print pca.replaceNanWithMean()
dataMat = pca.loadDataSet('testSet.txt')
print dataMat
print pca.pca(dataMat,1)
