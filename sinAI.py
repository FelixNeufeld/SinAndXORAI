import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Initiating the weights as random numbers between -1 and 1
inputWeights = [random.uniform(-1, 1) for _ in range(3)]
outputWeights = [random.uniform(-1, 1) for _ in range(3)]

#Error function
errorFunction = []

# Creating the teaching samples from 0 to 7 (aprox. 2 pi) in 0.1 steps
teachingSamples = [None] * 70

# Define learning rate

learningRate = 0.1

# Bias
bias = [random.uniform(-1,1) for _ in range(3)]

# Defining the Sigmoid function, as activation function
def sigmoid(v, c=1):
    return 1/(1+math.exp(-c*v))

# Defining the derivative of the Sigmoid function, for the backpropagation
def sigmoidDerivative(v, c=1):
    return sigmoid(v, c)*(1-sigmoid(v, c))

def calculateError(inputValue, outputValue, addToErrorFunction=False):
    error = math.pow(outputValue - math.sin(inputValue), 2)
    if(addToErrorFunction):
        errorFunction.append(error)
    return error

def trainNetwork(epochCount):
    for _ in range(epochCount):
        count = 1
        for inputValue in range(teachingSamples):
            hiddenLayer = [sigmoid(teachingSamples[inputValue] * inputWeights[j] + bias[j]) for j in range(len(inputWeights))]
            resultLayer = sum(hiddenLayer[i] * outputWeights[i] for i in range(len(outputWeights)))

            error = calculateError(inputValue, resultLayer, count == 69)

        # TODO: Gewichte und Bias anpassen
        count += 1
        

def neuralNetwork(inputValue):
    hiddenLayer = []
    for i in range(len(inputWeights)):
        hiddenLayer.append(sigmoid(inputValue * inputWeights[i] + bias[i]))

    resultLayer = sum(hiddenLayer[i] * outputWeights[i] for i in range(3))

    return resultLayer

def calculateSin():
    sin = [0] * 1000
    for i in range(1000):
        x = i/145
        sin[i] = neuralNetwork(x)
    return sin

def calculateRealSin():
    sin = [0] * 1000
    for i in range(1000):
        x = i/145
        sin[i] = math.sin(x)
    return sin


# Training
for i in range(70):
    teachingSamples = [random.uniform(0, 7) for _ in range(70)]

trainNetwork(100)

# Calculating the sinus
sin = calculateSin()
realSin = calculateRealSin()

#Plotting the result
plt.figure(1)
x_values = np.arange(0, 1000) / 145.0
plt.plot(x_values, sin, label='AI Sinus')
plt.plot(x_values, realSin, label='Real Sinus')
plt.xlabel('X-Achse')
plt.ylabel('Y-Achse')
plt.title('Calculated Values from the AI')
plt.legend()

# Showing Error Curve
plt.figure(2)
epochCount = list(range(1, 101))
plt.plot(epochCount, errorFunction, label='Errors')
plt.xlabel('X-Achse')
plt.ylabel('Y-Achse')
plt.title('Errors')
plt.legend()

# Showing the figures
plt.show()