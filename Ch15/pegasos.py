'''
Created on Feb 24, 2011
Sequential Pegasos 
the input T is k*T in Batch Pegasos
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):
    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        #dataArr.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
        dataArr.append([float(lineArr[0]), float(lineArr[1])])
        labelArr.append(float(lineArr[2]))
    return dataArr,labelArr

def seqPegasos(dataArray, labelArr, lam, T):
    m,n = shape(dataArray); w = zeros(n)
    for t in range(1, T+1):
        i = random.randint(m)
        eta = 1.0/(lam*t)
        p = predict(w, dataArray[i])
        if labelArr[i]*p < 1:
            w = (1.0 - 1/t)*w + eta*labelArr[i]*dataArray[i]
        else:
            w = (1.0 - 1/t)*w
        #print w
    return w
        
def predict(w, x):
    return vdot(w, x)

def batchPegasos(dataArray, labelArr, lam, T, k):
    m,n = shape(dataArray); w = zeros(n); 
    dataIndex = range(m)
    for t in range(1, T+1):
        wDelta = zeros(n) #reset wDelta
        eta = 1.0/(lam*t)
        random.shuffle(dataIndex)
        for j in range(k):#go over training set 
            i = dataIndex[j]
            p = predict(w, dataArray[i])        #mapper code
            if labelArr[i]*p < 1:                 #mapper code
                wDelta += labelArr[i]*dataArray[i] #accumulate changes  
        w = (1.0 - 1/t)*w + (eta/k)*wDelta       #apply changes at each T
    return w

def plotWs(finalWs, dataArray, labelArr):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1=[]; y1=[]; xm1=[]; ym1=[]
    for i in range(len(labelArr)):
        if labelArr[i] == 1.0:
            x1.append(dataArray[i,0]); y1.append(dataArray[i,1])
        else:
            xm1.append(dataArray[i,0]); ym1.append(dataArray[i,1])
    ax.scatter(x1, y1, marker='s', s=90)
    ax.scatter(xm1, ym1, marker='o', s=50, c='red')
    x = arange(-6.0, 8.0, 0.1)
    y = (-finalWs[0]*x - 0)/finalWs[1]
    #y2 = (0.43799*x)/0.12316
    y2 = (0.498442*x)/0.092387 #2 iterations
    ax.plot(x,y)
    ax.plot(x,y2,'g-.')
    ax.axis([-6,8,-4,5])
    ax.legend(('50 Iterations', '2 Iterations') )
    plt.show()

def main():
    datArr,labelArr = loadDataSet('testSet.txt')
    dataArray = array(datArr)
    finalWs = seqPegasos(dataArray, labelArr, 2, 5000)
    print finalWs
    plotWs(finalWs, dataArray, labelArr)
    finalWs = batchPegasos(dataArray, labelArr, 2, 50, 100)
    print finalWs
    plotWs(finalWs, dataArray, labelArr)

if __name__ == "__main__":
    main()
