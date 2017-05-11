import numpy as np
import os
import MykNN
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(linestr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMatrix = np.zeros((m, 32*32))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMatrix[i,:] = img2vector("digits/trainingDigits/{}".format(fileNameStr))
    testFileList = os.listdir("digits/testDigits")
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("digits/testDigits/{}".format(fileNameStr))
        classifierResult = MykNN.MykNNClassifier(vectorUnderTest, trainingMatrix, hwLabels, 3)
        print ("Real Answer VS Classified Answer: {} VS {}".format(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
    print ("The total number of errors is: {}".format(errorCount))
    print ("Error rate: {0:.2f}\%".format(errorCount * 100 / mTest))
