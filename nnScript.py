import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

iteration = 1
matrix_validation_label = np.array([])

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W


def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
       # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-1.0 * z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    global matrix_validation_label

    mat = loadmat('./mnist_all.mat') #loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data
    RANDOM_SIZE = 1000

    # Pick number of output nodes
    OUTPUT_NODES = 10

    A = mat.get('train0')
    B = mat.get('test0')

    a = range(A.shape[0])
    aperm = np.random.permutation(a)

    # Pick random samples
    A1 = A[aperm[0:RANDOM_SIZE],:]

    # Rest of the samples
    A2 = A[aperm[RANDOM_SIZE:],:]

    # Validation samples
    validation_data = np.array(A1, dtype=float)
    validation_label = np.array([0,] * validation_data.shape[0], dtype=np.int)

    # Training samples
    train_data = np.array(A2, dtype=float)
    train_label = np.array([0,] * train_data.shape[0], dtype=np.int)

    # Test data samples
    test_data = np.array(B, dtype=float)
    test_label = np.array([0,] * test_data.shape[0], dtype=np.int)

    matrix_validation_label = np.zeros((A2.shape[0], 10), dtype=np.int)
    matrix_validation_label[:,0] = 1

    # Populate data for rest of the nodes
    for i in range(1, OUTPUT_NODES):
        A = mat.get('train' + str(i))
        B = mat.get('test' + str(i))

        a = range(A.shape[0])
        aperm = np.random.permutation(a)

        A1 = A[aperm[0:RANDOM_SIZE],:]
        A2 = A[aperm[RANDOM_SIZE:],:]

        train_data = np.vstack((train_data, np.array(A2)))
        validation_data = np.vstack((validation_data, np.array(A1)))
        test_data = np.vstack((test_data, np.array(B)))

        temp_validation_label = np.array([i,] * A1.shape[0], dtype=np.int)
        validation_label = np.hstack((validation_label, temp_validation_label))

        temp_test_label = np.array([i,] * B.shape[0], dtype=np.int)
        test_label = np.hstack((test_label, temp_test_label))

        temp_train_label = np.array([i,] * A2.shape[0], dtype=np.int)
        train_label = np.hstack((train_label, temp_train_label))

        temp_matrix_validation_label = np.zeros((A2.shape[0], 10), dtype=np.int)
        temp_matrix_validation_label[:,i] = 1
        matrix_validation_label = np.vstack((matrix_validation_label, temp_matrix_validation_label))

    # Normalization
    train_data /= 255
    validation_data /= 255
    test_data /= 255

    # Feature Selection

    redundant_features = []
    redundant_features_counter = 0
    for col in train_data.T:
        if np.sum(col) == 0:
            redundant_features.append(redundant_features_counter)
        redundant_features_counter += 1

    redundant_features_validation = []
    redundant_features_counter = 0
    for col in validation_data.T:
        if np.sum(col) == 0:
            redundant_features_validation.append(redundant_features_counter)
        redundant_features_counter += 1

    redundant_features_test = []
    redundant_features_counter = 0
    for col in train_data.T:
        if np.sum(col) == 0:
            redundant_features_test.append(redundant_features_counter)
        redundant_features_counter += 1

    redundant_features_indices = np.intersect1d(np.intersect1d(redundant_features,redundant_features_validation), redundant_features_test)

    train_data = np.delete(train_data,redundant_features_indices,1)
    validation_data = np.delete(validation_data,redundant_features_indices,1)
    test_data = np.delete(test_data,redundant_features_indices,1)

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    global iteration
    print(iteration)
    iteration += 1

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0

    # Add Bias
    input_bias = np.ones((training_data.shape[0], 1), dtype=np.int)
    training_data = np.hstack((training_data, input_bias))

    # Hidden Layer net value
    net1 = np.dot(training_data, np.transpose(w1))

    # Output at hidden node
    out1 = sigmoid(net1)

    # Hidden Layer output
    hidden_bias = np.ones((out1.shape[0], 1))

    # Hidden Bias
    out1 = np.hstack((out1, hidden_bias))

    # Final output
    out2 = sigmoid(np.dot(out1, np.transpose(w2)))

    yl = matrix_validation_label

    delta_l = (yl - out2) * (1 - out2) * out2

    # Derivative W2
    err_fun_deriv_w2 = -np.dot(np.transpose(delta_l), out1)

    deriv_w1_param_1 = np.dot(delta_l, w2)

    # eq12 first half(multiplying Zj part)
    deriv_w1_param_2 = -(1-out1)*out1

    # eq12 multiplying above two
    deriv_w1_param_3 = deriv_w1_param_2*deriv_w1_param_1

    # eq12 multiplying with training data -- 785 * 4
    err_fun_deriv_w1 = np.dot(np.transpose(deriv_w1_param_3), training_data)

    # Derivative W1
    err_fun_deriv_w1 = err_fun_deriv_w1[0:n_hidden,:]

    err_fn_scalar = np.sum(np.square(yl - out2))/2

    err_fn_scalar_normalized = err_fn_scalar/training_data.shape[0]

    # Regularization in Neural Network

    reg_err_func_param1 = (lambdaval / (2*len(training_data))) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    reg_err_func_final = err_fn_scalar_normalized + reg_err_func_param1

    err_fun_deriv_w2 = (err_fun_deriv_w2 + lambdaval*w2) / training_data.shape[0]
    err_fun_deriv_w1 = (err_fun_deriv_w1 + lambdaval*w1) / training_data.shape[0]

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array

    obj_grad = np.concatenate((err_fun_deriv_w1.flatten(), err_fun_deriv_w2.flatten()), 0)

    return reg_err_func_final, obj_grad


def nnPredict(w1,w2,data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])

    data_bias = np.ones((data.shape[0], 1))

    # Hidden Bias
    data = np.hstack((data, data_bias))

    net1 = np.dot(data, np.transpose(w1))

    # Hidden Layer output
    out1 = sigmoid(net1)

    out1_bias = np.ones((out1.shape[0], 1))
    out1 = np.hstack((out1, out1_bias))

    # Final output
    labels = sigmoid(np.dot(out1, np.transpose(w2)))

    # The prediction is the index of the output unit with the max o/p
    labels = np.argmax(labels, axis=1)

    return labels

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.7;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
