import csv
from matplotlib import pyplot as plt
import numpy as np

# Implementation of logistic regression

# Computing loss function
def loss(y, yhat):
    m = y.shape[1]
    res = -y * np.log(yhat) - (1-y) * np.log(1-yhat)
    return np.sum(res) / m

# Computing gradient of the loss function
def gradient(activation, y, x):
    # activation = [1x100]
    # y = [1x100]
    # x = [100x2]
    # returns [1x2] gradient value for each weight that multiplies each feature
    m = y.shape[1]
    assert(activation.shape[0] == y.shape[0])
    assert (activation.shape[1] == y.shape[1])
    dz = activation - y
    assert(dz.shape[1] == x.shape[0])
    return np.dot(dz, x) / m


# Computing the logistic function
def logistic(z):
    return 1 / (1 + np.exp(-z))

# Input X,Y defining the features and the labels
input = []
with open('ex2data1.txt', 'r') as input_csv:
    reader = csv.reader(input_csv)
    for row in reader:
        input.append([float(i) for i in row])

input_csv.close()

#number of training samples
m = 75
num_features = 2

# X = [m,2]
X = np.array([[row[i] for i in range(num_features)] for row in input[:m]])
# Y = [1, 100]
Y = np.array([row[2] for row in input[:m]]).reshape(1,m)

x1 = [row[0] for row in X]
x2 = [row[1] for row in X]

# Visualizing data
colors = ['red' if i == 0 else 'blue' for i in Y[0]]
plt.figure(1)
plt.subplot(2,1,1)
plt.scatter(x1,x2, color=colors)

# Initialization of parameters theta(weights) and learning rate
theta = np.random.randn(num_features,1) * 0.01
print(theta.shape, X.shape)
b = 0
learning_rate = 0.0001
num_iter = 100

loss_vals = []
loss_val_it = []

for iteration in range(500):
    # yhat =  [2,1].T dot [m,2] = [
    yhat = logistic( np.dot(theta.T, X.T) ) # [1,100]
    loss_vals.append(loss(Y, yhat))
    plt.subplot(2,1,2)
    loss_val_it.append(iteration)
    plt.plot(loss_val_it, loss_vals)

    z = np.dot(theta.T, X.T) + b
    activation = logistic(z)

    dz = activation-Y
    dtheta = gradient(activation, Y, X)

    # gradient descent
    theta = theta - (learning_rate * dtheta).T
    b = b - learning_rate * np.sum(dz)/m

    # update loss function plot with new theta and iteration number

print(loss_vals[-1])
plt.subplot(2,1,1)
x1_vals = np.array([30,100])
x2_vals = (theta[1][0] * x1_vals + b)/ -theta[0][0]
plt.plot(x1_vals, x2_vals)
plt.show()