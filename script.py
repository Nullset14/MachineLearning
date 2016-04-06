import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 

    means = []
    class_labels = np.unique(np.array(y))

    for class_label in class_labels:
        label_indices, columns = np.where(y == class_label)
        means.append(np.mean(X[label_indices], axis=0))

    means = np.transpose(means)
    covmat = np.cov(X.T)

    return means, covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    means, covmats = [], []
    class_labels = np.unique(np.array(y))

    for class_label in class_labels:
        label_indices, columns = np.where(y == class_label)
        means.append(np.mean(X[label_indices], axis=0))

    means = np.transpose(means)

    for class_label in class_labels:
        label_indices, columns = np.where(y == class_label)
        covmats.append(np.cov(X[label_indices].T))

    return means, covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    acc = 0
    ypred = []
    for i in range(0, len(Xtest)):
        y = []
        for j in range(0, means.shape[1]):
            y.append(np.exp(-0.5 * np.dot(Xtest[i] - means.T[j], np.dot(np.linalg.inv(covmat), (Xtest[i] - means.T[j])))) * \
            1/(np.sqrt(2 * np.pi) ** Xtest.shape[1] * np.sqrt(np.linalg.det(covmat))))

        ypred.append(np.argmax(y, 0) + 1)

        if ypred[-1] == ytest[i][0]:
            acc += 1

    acc = 100 * acc/len(ytest)
    ypred = np.array(ypred)

    return acc, ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    acc = 0.0
    ypred = []
    for i in range(0, len(Xtest)):
        y = []
        for j in range(0, means.shape[1]):
            y.append(
                np.exp(-0.5 * np.dot(Xtest[i] - means.T[j], np.dot(np.linalg.inv(covmats[j]), (Xtest[i] - means.T[j])))) * \
                1 / (np.sqrt(2 * np.pi) ** Xtest.shape[1] * np.sqrt(np.linalg.det(covmats[j]))))

        ypred.append(np.argmax(y, 0) + 1)

        if ypred[-1] == ytest[i][0]:
            acc += 1

    acc = 100 * acc/len(ytest)
    return acc, np.array(ypred)

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1
    w = np.linalg.solve(np.dot(np.transpose(X), X), np.dot(np.transpose(X), y))
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    I = np.identity(X.shape[1])
    w = np.linalg.solve(np.add(np.dot(np.transpose(X), X), lambd * I), np.dot(np.transpose(X), y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    n1 = Xtest.shape[0]
    rmse = sqrt(1/n1 * np.dot(np.transpose(ytest - np.dot(Xtest, w)), ytest - np.dot(Xtest, w)))
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # error_grad is a 1d vector

    w = np.row_stack(np.array(w).flatten())

    x1 = np.dot(X, w)
    diff = np.subtract(y, x1)

    x = np.dot(np.transpose(diff), diff)
    y1 = lambd * np.dot(np.transpose(w), w)
    error = 0.5 * (np.add(x, y1))

    param_XTX = np.dot(np.transpose(X), X)
    param_XTY = np.dot(np.transpose(X), y)

    error_grad = np.subtract(np.dot(param_XTX, w), param_XTY) + lambd * w
    return error, error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))
    xd = np.ones((x.shape[0], 1))
    for pow in range(1, p+1):
        xi = np.reshape(np.power(x, pow), (x.shape[0], 1))
        xd = np.hstack((xd, xi))
    return xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
ldaacc = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA
means, covmats = qdaLearn(X, y)
qdaacc = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5,  20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0]*x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])))
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.show()

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])))
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.show()

# Problem 2

if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3_test = np.zeros((k, 1))
rmses3_train = np.zeros((k, 1))

for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    rmses3_test[i] = testOLERegression(w_l, Xtest_i, ytest)
    rmses3_train[i] = testOLERegression(w_l, X_i, y)
    i = i + 1


plt.plot(lambdas, rmses3_test)
plt.plot(lambdas, rmses3_train)
plt.legend(('Testing Data', 'Training Data'))
plt.show()

# Problem 4

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k, 1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='BFGS', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    rmses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
plt.plot(lambdas, rmses4)
plt.show()

# Problem 5

pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    rmses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    rmses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)
plt.plot(range(pmax), rmses5)
plt.legend(('No Regularization', 'Regularization'))
plt.show()
