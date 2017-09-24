import pca
print pca.replaceNanWithMean()
dataArray = pca.loadDataSet('testSet.txt')
print dataArray
print pca.pca(dataArray,1)
