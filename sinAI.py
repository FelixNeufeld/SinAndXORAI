import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Initiating the weights as random numbers between -1 and 1
inputWeights = [random.uniform(-1, 1) for _ in range(3)]
outputWeights = [[random.uniform(-1, 1)] for _ in range(3)]

#Error function
errorFunction = []

# Creating the teaching samples from 0 to 7 (aprox. 2 pi) in 0.1 steps
teachingSamples = [i/1000 for i in range(7000)]

# Creating the test samples to show the calculation of the sin function
testValues = [(i+0.05)/1000 for i in range(7000)]

# Define learning rate
learningRate = 0.1

#Defining epoche count
epocheCount = 2000

# Bias
bias = [random.uniform(-1,1) for _ in range(3)]
outputBias = random.uniform(-1,1)

# Defining the Sigmoid function, as activation function
def sigmoid(v, c=1):
    return 1/(1+math.exp(-c*v))

#Variating the learning data to speed up learning
def variateLearningData():
    for i in range(len(teachingSamples)):
        randomVariation = random.uniform(-0.01, 0.01)
        teachingSamples[i] = (i/1000) + randomVariation


def debugOutputLearning(count, diff, resultLayer):
    print("Epoche: " + str(count))
    print("diff: " + str(diff))
    print("input weights: " + str(inputWeights))
    print("output weights: " + str(outputWeights))
    print("bias: " + str(bias))
    print("output bias: " + str(outputBias))
    print("result: " + str(resultLayer))

def debugOutputNeuralNetwork(inputValue, i):
    print("InputValue: " + str(inputValue))
    print("input weight: " + str(inputWeights[i]))
    print("bias: " + str(bias[i]))
    print("Sigmoid v: " + str(inputValue * inputWeights[i] + bias[i]))

def trainNetwork(epochCount):
    global outputBias
    for count in range(epochCount):
        for inputValue in range(len(teachingSamples)):
            hiddenLayer = [sigmoid(teachingSamples[inputValue] * inputWeights[j] + bias[j]) for j in range(len(inputWeights))]
            resultLayer = 0
            for i in range(len(outputWeights)):
                resultLayer += hiddenLayer[i] * outputWeights[i][0]
            resultLayer += outputBias

            # error = calculateError(teachingSamples[inputValue], resultLayer, count == 1)
            diff = math.sin(teachingSamples[inputValue])-resultLayer
            if(inputValue <= 0):
                errorFunction.append(abs(diff))

            for j in range(len(inputWeights)):
                inputWeights[j] += -learningRate * (-2*(diff)*outputWeights[j][0]*hiddenLayer[j]*(1-hiddenLayer[j])*teachingSamples[inputValue])
                bias[j] += -learningRate * (-2*(diff)*outputWeights[j][0]*hiddenLayer[j]*(1-hiddenLayer[j]))
            for j in range(len(outputWeights)):
                outputWeights[j][0] += -learningRate * (-2*(diff)*hiddenLayer[j])
                outputBias += (-learningRate * (-2*(diff)))
        variateLearningData()
        if(count % 100 == 0):
            print(str((count/epochCount)*100) + "%")



            
def neuralNetwork(inputValue):
    hiddenLayer = []
    for i in range(len(inputWeights)):
        hiddenLayer.append(sigmoid(inputValue * inputWeights[i] + bias[i]))

    resultLayer = sum(hiddenLayer[i] * outputWeights[i][0] for i in range(3))
    resultLayer += outputBias

    return resultLayer

def calculateSin():
    sin = [0] * len(testValues)
    for i in range(len(testValues)):
        sin[i] = neuralNetwork(testValues[i])
    return sin

def calculateRealSin():
    sin = [0] * len(testValues)
    for i in range(len(testValues)):
        sin[i] = math.sin(testValues[i])
    return sin


# Training
trainNetwork(epocheCount)

# Calculating the sinus
sin = calculateSin()
realSin = calculateRealSin()

#Plotting the result
plt.figure(1)
plt.plot(testValues, sin, label='AI Sinus')
plt.plot(testValues, realSin, label='Real Sinus')
plt.xlabel('X-Achse')
plt.ylabel('Y-Achse')
plt.title('Calculated Values from the AI')
plt.legend()

# Showing Error Curve
plt.figure(2)
epochCount = list(range(0, epocheCount))
plt.plot(epochCount, errorFunction, label='Errors')
plt.xlabel('X-Achse')
plt.ylabel('Y-Achse')
plt.title('Errors')
plt.legend()

# Showing the figures
plt.show()