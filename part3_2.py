#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 06:20:34 2019

@author: vanshsmacpro
"""

import numpy as np 
import sys
import timeit


# function to calculate lambda

def lambdaa(train_data , train_label):
    
    alpha = 5
    beta = 10
    phi  = train_data
    t = train_label
    phi  = np.array(phi)
    t = np.array(t)
    phi_t = phi.T
    I = np.identity(len(phi[0]), dtype = float)
    phi_t_t = phi_t.dot(t)
    phi_t_phi = phi_t.dot(phi)
    
    for i in range(100):
        A =  phi_t_phi.dot(beta) 
        B = I.dot(alpha)
        S_N_inverse = np.add(A, B) 
        S_N = np.linalg.inv(S_N_inverse)
        S_N = np.array(S_N)
        
        M_N_beta  = S_N.dot(phi_t_t)
        M_N = M_N_beta.dot(beta)
        M_N = np.array(M_N)
        
        gamma = gamma_value(A , alpha) 
        #len(gamma)
        alpha = calculate_alpha(gamma, M_N)
        beta = calculate_beta(gamma , t , M_N , phi )
        
    lambdaa = alpha/beta
    return lambdaa

# function to calculate alpha
       
def calculate_alpha(gamma, M_N):
    M_N_t = M_N.T
    M_sqaure = M_N_t.dot(M_N)
    
    alpha = gamma/M_sqaure
    
    return alpha
# function to calculate beta

def calculate_beta(gamma , t , M_N , phi ):
    
    summ = 0 
    
    for i in range(len(phi)):
        t_i = t[i]
        m_n = M_N
        m_n_t = m_n.T
        phi_i = phi[i]
        #print(m_n_t)
        m_n_t_phi = m_n_t.dot(phi_i)
        m_n_t_ph = t_i - m_n_t_phi
        m_n_t_sqaure = np.square(m_n_t_ph)
        summ = summ + m_n_t_sqaure
    N = len(phi)
    X = N - gamma
    beta = X/summ
    return beta
        
    
        
#function to calculate gamma function
        
def gamma_value(A,  alpha):
    
    eigenvalues , eigenvectors = np.linalg.eig(A)
    
    gamma  = 0
    summ = 0 
    for i in range(len(eigenvalues)):
        summ = (eigenvalues[i])/(alpha + (eigenvalues[i]))
        gamma = gamma + summ
        
    return gamma 

#calculating w from train data 
    
def training_set1(matrix_data,train_label_100_10,z):
    t = train_label_100_10

    t = t.reshape(t.shape[0],1)
#print(t)
    phi = matrix_data
    phi_t = phi.T
    B = phi_t.dot(phi)
    I = np.identity(len(phi[0]), dtype = float)

#for i in range(151):
 #   print(i)
    
   
    lambdaa = z
    A = I.dot(lambdaa)
    inverse = np.add(A,B)
    after_inverse = np.linalg.inv(inverse)
    
    t_phi = phi_t.dot(t)
    w = after_inverse.dot(t_phi)
    
    
    return w 

# calculating MSE from test data
    
def test_set1(test_100_10_data,test_100_10_label,w):
    t = test_100_10_label

    t = t.reshape(t.shape[0],1)
#print(t)
    phi = test_100_10_data
    
   
    MSE = 0
        
    for i in range(len(phi)):
        
        MSE_phi = phi[i]
        t_i = t[i]
        target_variable = t_i[0]
    
    
        pi_i = MSE_phi.dot(w)
        
        phi_w = pi_i
    
        phi_w = phi_w - target_variable
            
        phi_w = np.square(phi_w)
        #print(i)
    
        MSE = MSE + phi_w
   
    MSE = MSE/int(len(phi))
    
    
        

    return MSE
    
## main function
    
if __name__ == "__main__":
    
    start = timeit.default_timer()
    train_data = np.genfromtxt(sys.argv[1], delimiter=',')
    train_label = np.genfromtxt(sys.argv[2],delimiter = ",")
    
    
    test_100_10_data = np.genfromtxt(sys.argv[3], delimiter=',')
    test_100_10_label = np.genfromtxt(sys.argv[4], delimiter=',')
        
    len(train_data)
    
    #print(average_mse)
    
    lambdaa = lambdaa(train_data , train_label)
    
   
    print("here the lambda is  " , lambdaa)
    
    w  = training_set1(train_data, train_label , lambdaa)
    test_MSE = test_set1(test_100_10_data ,test_100_10_label, w )
    
    print("test MSE for the following lambda is ", test_MSE[0])
    
    
    
    
    end = timeit.default_timer()
    time = end - start
    print("running time is"  , time)
    
    
    