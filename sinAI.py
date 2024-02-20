import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Defining the Sigmoid function, as activation function
def sigmoid(v, c=1):
    return 1/(1+math.exp(-c*v))

# Defining the matrix multiplication
def matrixMultiplication(x, y):
    # Checking if the dimensions are fitting
    if(len(x[0]) != len(y)):
        print("Matrix x and y can not be multiplied")
        return None
    
    resultMatrix = [[0] * len(y[0]) for _ in range(len(x))]

    # Multiplying the matrices
    for i in range(len(x)):
        for j in range(len(y[0])):
            for k in range(len(y)):
                resultMatrix[i][j] += x[i][k] * y[k][j]
    return resultMatrix



# Initiating the weights as random numbers between -1 and 1
inputWeights = [random.uniform(-1,1) for _ in range(3)]
outputWeights = [random.uniform(-1,1) for _ in range(3)]

# Creating the teaching samples from 0 to 7 (aprox. 2 pi) in 0.1 steps
teachingSamples = [None] * 70

for i in range(70):
    teachingSamples[i] = i/10

def neuralNetwork(inputValue):
    hiddenLayer = [0] * 3
    for i in range(3):
        hiddenLayer[i] = inputValue * inputWeights[i]

    resultLayer = [0] * 3
    for i in range(3):
        resultLayer[i] = hiddenLayer[i] * outputWeights[i]

    sum = 0
    for i in range(3):
        sum += resultLayer[i]

    return sum

def calculateSin():
    sin = [0] * 1000
    for i in range(1000):
        x = i/145
        sin[i] = neuralNetwork(x)
    return sin

# Calculating the sinus
sin = calculateSin()
print(sin)
#Plotting the result
x_values = np.arange(0, 1000) / 145.0
plt.plot(x_values, sin, label='Sinus')
plt.xlabel('X-Achse')
plt.ylabel('Y-Achse')
plt.title('Calculated Values from the AI')

plt.legend()
plt.show()