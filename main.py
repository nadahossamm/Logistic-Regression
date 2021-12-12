import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_hypothesis(theta, x):
    z = np.dot(x, theta)
    h = 1 / (1 + np.exp(-z))
    return h


def calc_cost_function(theta, x, y):
    h = calc_hypothesis(theta, x)
    f = y * np.log(h)
    s = (1 - y) * np.log(1 - h)

    return (-1 / (y.size)) * sum(f + s)


def gradient_decent(theta, x, y,alpha):
    Range = 100
    error = []
    cost = calc_cost_function(theta, x, y)
    for i in range(Range):
        oldcost = cost
        error.append(oldcost)
        h_x = calc_hypothesis(theta, x)
        theta = theta + (alpha * np.dot((y - h_x).T, x))
        cost = calc_cost_function(theta, x, y)
    return theta, error


def predict(x, theta):
    h = calc_hypothesis(theta, x)
    pred_value = np.where(h >= .5, 1, 0)
    return pred_value


def calc_accuricy(theta, x, y):
    y_prdicet = predict(x, theta)
    acc = np.sum(np.equal(y, y_prdicet)) / len(y)
    return acc


# Reading From File
dataset = pd.read_csv("heart.csv")
Y = dataset['target']
X = dataset[['trestbps', 'chol', 'thalach', 'oldpeak']]

# Normalization
X_normalized = (X - X.mean()) / (X.std())
X = X_normalized

X.insert(0, 'one', 1)
X = np.array(X)
Y = np.array(Y).flatten()

# devide data to train ans test

trainSize = int(Y.size * .8)

XTrain = X[:trainSize]
XTest = X[trainSize:]

YTrain = Y[:trainSize]
YTest = Y[trainSize:]


#[0.001,0.003,0.01,0.03,0.1,0.3,0.5,1]
alphas=[0.001,0.003,0.01,0.03]

for i in alphas :
    theta = np.array([0, 0, 0, 0, 0])
    theta2, error = gradient_decent(theta, XTrain, YTrain,i)
    print('alpha = ' , i )
    print("accuracy of training ", calc_accuricy(theta2, XTrain, YTrain))
    print("accuracy of testing  ", calc_accuricy(theta2, XTest, YTest))
    print('-------------------------------------')
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration')
    plt.plot(error, c='blue', label='Cost Function')
    plt.legend()
    plt.show()


