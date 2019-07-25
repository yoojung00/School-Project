#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from Activation import Activation
from HiddenLayer import HiddenLayer

class MLP:
    """
    """      
    def __init__(self, layers, activation=[None,'relu','relu','softmax']):
        """
        :param layers: A list containing the number of units in each layer.
        
        :param activation: The activation function to be used. Can be "logistic", "tanh","leakyrelu", "relu" and "softmax".
        """        
        # Initialize layers and build multilayer neural network (MNN).
        self.layers=[]
        self.params=[]
        
        self.activation=activation
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))
            
    def forward(self, input, p, batch_norm):
        """
        Forward function of MNN.
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        
        :type p: float, range of (0, 1]
        :param p: the probability of neuron to be active at each layer, p is (0, 1]
        
        :type batch_norm: bool
        :param batch_norm: bool parameter, True is to do batch normalization, False is not to do.
        """
        for layer in self.layers:
            output=layer.forward(input, p, batch_norm)
            # No dropout and batch normalization in output layer.
            if layer is self.layers[-1]:
                output = layer.forward(input, p=1, batch_norm=False)
            input=output
        return output
    
    def forward_predict(self,input):
        """
        Forward function of MNN used in prediction.
        """
        for layer in self.layers:
            output=layer.forward_predict(input)
            input=output
        return output
        
    def criterion_MSE(self,y,y_hat):
        """
        Mean squared error loss function.
        """
        activation_deriv=Activation(self.activation[-1]).f_deriv
        error = y-y_hat
        loss = np.sum(error**2)
        # calculate the delta of the output layer
        delta=-error*activation_deriv(y_hat)/y.shape[0]    
        
        return loss,delta
    
    def criterion_cross_entropy(self, y, y_hat):
        """
        Softmax and cross-entropy loss function.
        :type y: numpy.array
        :param y: true output.
        
        :type y_hat: numpy.array
        :param y_hat: predicted output.
        """
        m = y.shape[0]
        loss = 0 - np.sum(y * np.log(y_hat+1e-10), axis=1, keepdims=True)
        delta = (y_hat - y)/m
        return loss, delta
        
    def backward(self, delta):
        """
        Backward function for MNN.
        :type delta: numpy.array
        :param delta: the derivative output from last layer.
        """ 
        delta=self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)
            
    def momentum_update(self, lr, lb, r, V_dw, V_db):
        """
        Momentum optimazation method for updating weight and bias.
        :type lr: float
        :param lr: learning rate.
        
        :type lb: float
        :param lb: weight decay parameter.
        
        :type lb: float, used to be 0.9
        :param lb: momentum term.
        
        :type V_dw: list
        :param V_dw: record the gradient descent of weight.
        
        :type V_db: list
        :param V_db: record the gradient descent of bias.
        """
        for i in range(0, len(self.layers)-1):
            layer = self.layers[i]
            V_dw[i] = r * V_dw[i] + lr * layer.grad_W
            V_db[i] = r * V_db[i] + lr * layer.grad_b

            layer.W -= (V_dw[i] + lr * lb * layer.W)
            layer.b -= V_db[i]
            
            if layer.batch_norm is True:
                layer.gamma -= lr * layer.grad_gamma
                layer.beta -= lr * layer.grad_beta
            
    def update(self,lr,lb):
        """
        Update function for weight, bias, gamma and beta.
        
        :type lr: float
        :param lr: learning rate.
        
        :type lb: float
        :param lb: weight decay parameter.
        """
        for layer in self.layers:
            # weight decay in updating weight.
            layer.W -= (lr * layer.grad_W + lr*lb*layer.W)
            layer.b -= lr * layer.grad_b
            if layer.batch_norm is True:
                layer.gamma -= lr * layer.grad_gamma
                layer.beta -= lr * layer.grad_beta
            
    def get_minibatches(self, X, y, minibatches_size):
        """
        This function is to generate mini-batches dataset.
        :type X: numpy.array
        :param X: input data.
        
        :type y: numpy.array
        :param y: true output.
        """

        n = X.shape[0]
        mini_batches = []

        rand_idx = list(np.random.permutation(n))
        rand_X = X[rand_idx, :]
        rand_y = y[rand_idx, :]

        m = n // minibatches_size

        for i in range(m):
            X_mini_batch = rand_X[i * minibatches_size: (i + 1) * minibatches_size, :]
            y_mini_batch = rand_y[i * minibatches_size: (i + 1) * minibatches_size, :]
            mini_batches.append((X_mini_batch, y_mini_batch))

        if n % minibatches_size != 0:
            X_mini_batch = rand_X[m * minibatches_size:, :]
            y_mini_batch = rand_y[m * minibatches_size:, :]
            mini_batches.append((X_mini_batch, y_mini_batch))
            
        return mini_batches

    def fit(self,X,y,learning_rate, epochs, minibatches_size, loss_function, lb, p, batch_norm, momentum, r):
        """
        Online learning.
        :param X: Input data or features.
        :param y: Input targets.
        :param learning_rate: parameters defining the speed of learning.
        :param epochs: number of times the dataset is presented to the network for learning.
        :param minibatches_size: the size of mini-batches.
        :param minibatches_size: 'MSE' or 'cross_entropy'.
        :param lb: weight decay parameter.
        :param p: the probability of active neurons in each layer.
        :param batch_norm: True for applying batch normalization, False for not applying.
        :param momentum: True for applying momentum, False for not applying.
        :param r: momentum parameter.
        """ 
        X=np.array(X)
        y=np.array(y)
        to_return = np.zeros(epochs)
        
        for k in tqdm(range(epochs)):
            
            # Initialize parameters for momentum.
            V_dw = []
            V_db = []
            for layer in self.layers:
                v_w = np.zeros_like(layer.grad_W)
                v_b = np.zeros_like(layer.grad_b)
                V_dw.append(v_w)
                V_db.append(v_b)
                
            # Using stochastic gradient descent    
            if minibatches_size == 0:
                loss=np.zeros(X.shape[0])
                for it in range(X.shape[0]):
                    i=np.random.randint(X.shape[0])
                    # forward pass
                    y_hat = self.forward(X[i], p, batch_norm)
                    # choose loss function
                    if loss_function == 'MSE':
                        loss[it],delta=self.criterion_MSE(y[i],y_hat)
                    elif loss_function == 'cross_entropy':
                        loss[it], delta = self.criterion_cross_entropy(y[i], y_hat)
                    # backward pass
                    self.backward(delta)
                    # update
                    if momentum is True:
                        self.momentum_update(learning_rate, lb, r, V_dw, V_db)
                    else:
                        self.update(learning_rate,lb)
                to_return[k] = np.mean(loss)
            
            # Using mini-batches.
            elif minibatches_size !=0 :
                loss = np.zeros(X.shape[0])
                mini_batches = self.get_minibatches(X, y, minibatches_size)

                for it, mini_batch in enumerate(mini_batches):
                    loss_it = 0
                    (mini_batch_X, mini_batch_y) = mini_batch
                    # forward pass
                    y_hat = self.forward(mini_batch_X, p, batch_norm)
                    # backward pass
                    if loss_function == 'MSE':
                        loss_it, delta = self.criterion_MSE(mini_batch_y, y_hat)
                    elif loss_function == 'cross_entropy':
                        loss_it, delta = self.criterion_cross_entropy(mini_batch_y, y_hat)
                    loss[it] = np.mean(loss_it)

                    self.backward(delta)
                    # update
                    if momentum is True:
                        self.momentum_update(learning_rate, lb, r, V_dw, V_db)
                    else:
                        self.update(learning_rate,lb)
                    
                to_return[k] = np.mean(loss)
                
            if batch_norm is True:
                for i in range(len(self.layers)):
                    if i != len(self.layers)-1:
                        self.layers[i].BN_prepare_predict(minibatches_size)
        return to_return

    def predict(self, x):
        x = np.array(x)
        output = np.zeros((x.shape[0], 10))
        for i in np.arange(x.shape[0]):
            output[i] = self.forward_predict(x[i,:])
    
        y_pred = np.argmax(output, axis = 1)
        return y_pred


