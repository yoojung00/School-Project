#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Activation import Activation
import numpy as np

class HiddenLayer(object):    
    def __init__(self,n_in, n_out,
                 activation_last_layer='softmax',activation='relu', W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is ReLU

        Hidden unit activation is given by: ReLU(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input=None
        # initialize activation
        self.activation=Activation(activation).f
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv
        
        # initialize weight and bias
        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        self.b = np.zeros(n_out,)
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        
        # initialize batch normalization parameters
        self.batch_norm = None
        self.x_norm = None
        self.gamma=np.ones(n_out,)
        self.beta=np.zeros(n_out,)
        self.grad_gamma=np.zeros(self.gamma.shape)
        self.grad_beta=np.zeros(self.beta.shape)
        self.BN_mean = []
        self.BN_var = []
        
        # initialize dropout parameters
        self.retain = None
        self.dropout_p = None                
        
    def batchnorm_forward(self, x, err=1e-8):
        """
        Batch normalization forward function
        :type x: numpy.array
        :param x: the forward output from linear function.
        """
        #reference: https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
        self.X = x
        self.mu = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.x_norm = (self.X - self.mu) / np.sqrt(self.var + err)
        out = self.x_norm * self.gamma + self.beta
        self.BN_mean.append(self.mu)
        self.BN_var.append(self.var)
        return out
    
    def batchnorm_backward(self, delta):
        """
        Batch normalization backward function
        :type delta: numpy.array
        :param delta: the derivative output from last layer.
        """
        #reference: https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
        X = self.X
        X_norm = self.x_norm
        mu = self.mu
        var = self.var
        gamma = self.gamma
        beta = self.beta
        
        N, D = X.shape

        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + 1e-8)

        dX_norm = delta * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(delta * X_norm, axis=0)
        dbeta = np.sum(delta, axis=0)

        self.grad_gamma = dgamma
        self.grad_beta = dbeta
        return dX
    
    def BN_prepare_predict(self, minibatches_size):
        """
        Calculate the mean and variance for doing batch normalization in predicting.
        :type minibatches_size: int
        :param minibatches_size: the size of mini-batches.
        """
        self.mu = np.mean(self.BN_mean,axis = 0)
        self.var = minibatches_size/(minibatches_size-1) * np.mean(self.BN_var, axis = 0)
        
    def BN_forward_predict(self, X):
        """
        Batch normalization forward function for predicting.
        :type x: numpy.array
        :param x: the forward output from linear function.
        """
        self.X = X
        self.X_norm = (X - self.mu)/np.sqrt(self.var + 1e-8)
        out =  self.X_norm*self.gamma + self.beta
        return out
    
    def dropout(self, x, p):
        """
        Dropout forward function
        :type p: float, range of (0, 1]
        :param p: the probability of neuron to be active at each layer.
        """
        # dropout forward function
        #reference: https://wiseodd.github.io/techblog/2016/06/25/dropout/
        self.retain = np.random.binomial(1, p, size=x.shape[1])/p
        out = x * self.retain
        return out
        
    def forward(self, input, p, batch_norm):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        
        :type p: float, range of (0, 1]
        :param p: the probability of neuron to be active at each layer, p is (0, 1]
        
        :type batch_norm: bool
        :param batch_norm: bool parameter, True is to do batch normalization, False is not to do.
        '''
        input = input.reshape(-1, self.W.shape[0])
            
        lin_output = np.dot(input, self.W) + self.b
        
        
        # activation
        act_output = (
            lin_output if self.activation is None
            else self.activation(lin_output))
        
        # batch normalization
        self.batch_norm = batch_norm
        if batch_norm is False:
            batch_output = act_output
        else:
            batch_output= self.batchnorm_forward(act_output)
        
        # dropout
        self.dropout_p = p
        dropout_output =(
            batch_output if p == 1
            else self.dropout(batch_output, p))
        
        self.output=dropout_output
        self.input=input
        return self.output
    
    
    def backward(self, delta, output_layer=False):
        """
        Backward function for each layer.
        :type delta: numpy.array
        :param delta: the derivative output from last layer.
        """         
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta, axis=0)
        if self.batch_norm is True:
            delta = self.batchnorm_backward(delta)
        if self.dropout_p != 1:
            delta *= self.retain
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        
        return delta
    
    
    def forward_predict(self, input):
        """
        Forward function for each layer in prediction part.
        :type input: numpy.array
        :param input: the output from last layer.
        """ 
        input = input.reshape(-1, self.W.shape[0])
        lin_output = np.dot(input, self.W) + self.b

            
        act_output = (
            lin_output if self.activation is None
            else self.activation(lin_output))
        
            # batchnorm forward for predicting
        
        batch_output = (
                self.BN_forward_predict(act_output) if self.batch_norm is True
                else act_output)
        
        self.output=batch_output
        self.input=input
        return self.output