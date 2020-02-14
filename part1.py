#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 06:13:23 2019

@author: vanshsmacpro
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

# function to calculate w and trainset MSE 

def training_set(train_100_10_data,train_label_100_10):
    t = train_label_100_10

    t = t.reshape(t.shape[0],1)
    phi = train_100_10_data
    phi_t = phi.T
    B = phi_t.dot(phi)
    I = np.identity(len(phi[0]), dtype = float)

    matrix = []
    matrix_for_w = []
    for j in range(151):
        lambdaa = j
        A = I.dot(lambdaa)
        inverse = np.add(A,B)
        after_inverse = np.linalg.inv(inverse)
    
        t_phi = phi_t.dot(t)
        w = after_inverse.dot(t_phi)
    
        matrix_for_w.append(w)
    
        MSE = 0 
        for i in range(len(phi)):
        
            MSE_phi = phi[i]
            t_i = t[i]
            target_variable = t_i[0]
    
    
            pi_i = MSE_phi.dot(w)
        
            phi_w = pi_i[0]
    
            phi_w = phi_w - target_variable
            
            phi_w = np.square(phi_w)
        #print(i)
    
            MSE = MSE + phi_w
   
        MSE = MSE/int(len(phi))
    
    
        matrix.append(MSE)

    return matrix , matrix_for_w

# function for calculating test MSE 

def test_set(test_100_10_data,test_100_10_label,matrix_for_w):
    t = test_100_10_label

    t = t.reshape(t.shape[0],1)
#print(t)
    phi = test_100_10_data
    
    matrix = []
    for j in range(151):
        w = matrix_for_w[j]
        MSE = 0
        
        for i in range(len(phi)):
        
            MSE_phi = phi[i]
            t_i = t[i]
            target_variable = t_i[0]
    
    
            pi_i = MSE_phi.dot(w)
        
            phi_w = pi_i[0]
    
            phi_w = phi_w - target_variable
            
            phi_w = np.square(phi_w)
        #print(i)
    
            MSE = MSE + phi_w
   
        MSE = MSE/int(len(phi))
    
    
        matrix.append(MSE)

    return matrix 

# plot graphs 

def plot_graphs(final_matrix , test_matrix):
    lambdaa = []
    np.shape(final_matrix)
    
    for i in range(151):
      
        lambdaa.append(i)
   
    plt.plot(lambdaa , final_matrix , 'b', label = "train data")
    plt.plot(lambdaa , test_matrix , 'r', label = "test data")
    plt.xlabel("lamdaa values")
    plt.ylabel("Mean square error")
    plt.legend()
    plt.show()


# main function 
    
if __name__ == "__main__":
    train_100_10_data = np.genfromtxt(sys.argv[1], delimiter=',')
    train_label_100_10 = np.genfromtxt(sys.argv[2],delimiter = ",")
    
    final_matrix, matrix_for_w = training_set(train_100_10_data,train_label_100_10)
    
    
    test_100_10_data = np.genfromtxt(sys.argv[3], delimiter=',')
    test_100_10_label = np.genfromtxt(sys.argv[4], delimiter=',')
    test_matrix = test_set(test_100_10_data,test_100_10_label,matrix_for_w)
    print(test_matrix)
    plot_graphs(final_matrix , test_matrix)