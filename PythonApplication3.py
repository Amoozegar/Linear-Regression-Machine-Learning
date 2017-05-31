import numpy

housingData = open("housing.data").readlines()
for i in range(len(housingData)):
    housingData[i] = housingData[i].rstrip()
    housingData[i] = housingData[i].split()
    for j in range(14):
        housingData[i][j] = float(housingData[i][j])


def OptimalWeightEstimation(XDataMatrixParam, rDataVectorParam):
    temp = numpy.dot( XDataMatrixParam.transpose(), XDataMatrixParam)
    temp2 = numpy.dot(numpy.linalg.inv(temp), XDataMatrixParam.transpose())
    returnValue = numpy.dot(temp2, rDataVectorParam)
    return returnValue

def  AssociatedValuePrediction(x, W):
    return numpy.dot(x, W)

def MeanSquareError(rArray, r_hatArray):
    MSE = 0.0
    for i in range(rArray.shape[0]):
        MSE += numpy.power((rArray[i,0] - r_hatArray[i,0]), 2)
    return (MSE / rArray.shape[0])
         
    
TrainingSizes = [200, 300, 400]
for trainingSize in TrainingSizes:
    XDataMatrixForTraining = numpy.matrix(housingData)[:trainingSize,:13]
    rDataVectorForTraining = numpy.matrix(housingData)[:trainingSize,13]
    W_hat = OptimalWeightEstimation(XDataMatrixForTraining, rDataVectorForTraining)    
    r_hatVector = numpy.matrix(numpy.zeros((len(housingData) - trainingSize, 1)))
    for j in range(len(housingData) - trainingSize):
        r_hatVector[j,0] = AssociatedValuePrediction(numpy.matrix(housingData)[trainingSize + j,:13], W_hat)
    print "MSE for training set size " + str(trainingSize) + " : " + str(MeanSquareError(numpy.matrix(housingData)[trainingSize + 1:,13], r_hatVector))

