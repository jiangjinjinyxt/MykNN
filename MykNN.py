import numpy as np
import operator

def file2matrix(fileName, *params):
    contents = open(fileName)
    contents = contents.readlines()
    rows = len(contents)
    cols = len(contents[0].split())
    #default: the last column is labels
    if len(params) == 0:
        returnMatrix = np.zeros((rows, cols - 1))
        returnLabels = [0] * rows
        for i in range(rows):
            currentLine = contents[i].split()
            returnMatrix[i,:] = currentLine[:-1]
            returnLabels[i] = currentLine[-1]
    elif len(params) == 1:
        returnMatrix = np.zeros((rows, params[0]))
        returnLabels = [] * rows
        for i in range(rows):
            currentLine = contents[i].split()
            returnMatrix[i,:] = currentLine[:params[0]]
            returnLabels[i] = currentLine[-1]
    elif len(params) > 1:
        returnMatrix = np.zeros((rows, params[0]))
        returnLabels = params[1]
        for i in range(rows):
            currentLine = contents[i].split()
            returnMatrix[i,:] = currentLine[:params[0]]
    else:
        return None,None
    return returnMatrix, returnLabels
def MykNNClassifier(testData, dataSet, dataLabels, k):
    diffMatrix = np.tile(testData, (dataSet.shape[0], 1)) - dataSet
    distances = ((diffMatrix ** 2).sum(axis = 1))**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        currentLabel = dataLabels[sortedDistIndices[i]]
        classCount[currentLabel] = classCount.get(currentLabel, 0) + 1
    return sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)[0][0]
def autoNorm(dataSet):
    minVals = dataSet.min(axis = 0)
    maxVals = dataSet.max(axis = 0)
    return (dataSet - np.tile(minVals, (dataSet.shape[0],1))) / (maxVals - minVals), minVals, maxVals
def datingClassTest(fileName, ratio = 0.1):
    #1 - ratio as training set, ratio as testing set
    datingDataMatrix, datingLabels = file2matrix(fileName)
    normMatrix, minVals, maxVals = autoNorm(datingDataMatrix)
    errorCount = 0
    lenTrainingSet = int((1-ratio)*datingDataMatrix.shape[0])
    for i in range(lenTrainingSet):
        result = MykNNClassifier(normMatrix[i,:],normMatrix[lenTrainingSet:,:],datingLabels[lenTrainingSet:],3)
        if datingLabels[i] != result:
            errorCount += 1
            print("Error-{}-- Real Answer VS Classified Answer: {} VS {}".format(errorCount,datingLabels[i], result))
    print ("Total Error Rate: {0:.2f}\%".format(errorCount*100/lenTrainingSet))
def classifyPerson():
    percentTats = float(input("percentage of time spent playing games: "))
    ffMiles = float(input("frequent of miles earned per year: "))
    iceCream = float(input("liters of ice cream consumed per year: "))
    datingDataMatrix, datingLabels = file2matrix("datingTestSet2.txt")
    normMatrix, minVals, maxVals = autoNorm(datingDataMatrix)
    result = MykNNClassifier((np.array([percentTats, ffMiles, iceCream])-minVals)/(maxVals-minVals), normMatrix, datingLabels, 3)
    print ("This person is classified as: {}".format(result))
