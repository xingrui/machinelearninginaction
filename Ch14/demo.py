import svdRec
from numpy import *
svdRec.imgCompress()
myArray = array(svdRec.loadExData())
myArray[0,1]=myArray[0,0]=myArray[1,0]=myArray[2,0]=4
myArray[3,3]=2
print myArray
print svdRec.recommend(myArray, 2)
print svdRec.recommend(myArray, 2, estMethod=svdRec.svdEst)
