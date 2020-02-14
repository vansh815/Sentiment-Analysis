#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 06:13:23 2019

@author: vanshsmacpro
"""


import numpy as np
import sys
import matplotlib.pyplot as plt

# function to calculate w for 3 values of lambda 

def training_set(matrix_data,matrix_label,z):
    matrix = []
    matrix_for_w = []
        
    len(matrix_data)
    len(matrix_data[0])
    len(matrix_label[0])
    
    for i in range(len(matrix_data)):
        t = matrix_label[i]
        phi = matrix_data[i]
        #np.shape(t)
        
        #print(t)
        len(t)
        #print(t)
        phi = np.array(phi)
        t = np.array(t)
        #print(phi)
        np.shape(phi)
        phi_t = phi.T
        B = phi_t.dot(phi)
        I = np.identity(len(phi[0]), dtype = float)


        
        lambdaa = z
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
            target_variable = t_i
    
    
            pi_i = MSE_phi.dot(w)
        
            phi_w = pi_i
    
            phi_w = phi_w - target_variable
            
            phi_w = np.square(phi_w)
        #print(i)
    
            MSE = MSE + phi_w
   
        MSE = MSE/int(len(phi))
    
    
        matrix.append(MSE)

    return matrix , matrix_for_w

# funciton to calculate MSE on test data

def test_set(test_100_10_data,test_100_10_label,matrix_for_w):
    t = test_100_10_label

    t = t.reshape(t.shape[0],1)
#print(t)
    phi = test_100_10_data
    
    matrix = []
    for j in range(100):
        w = matrix_for_w[j]
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
    
    
        matrix.append(MSE)

    return matrix 

# function to split data 

def split_data(train_100_10_data ,train_label_100_10 ):
    matrix_data = []
    matrix_label = []
    n = 10
    
    count = 0 
    while n <= 1000 : 
        traini = []
        testi = []
        for i in range(n):
            
            traini.append(train_100_10_data[count])
            testi.append(train_label_100_10[count])
            count = count + 1
            
            if count == (n-1):
                n = count + 11
                count = 0  
                
        matrix_data.append(traini)
        matrix_label.append(testi)
    return matrix_data , matrix_label

# function to plot graphs

def plot_graphs(final_matrix , test_matrix , z):
    lambdaa = []
    np.shape(final_matrix)
    #count = 0 
    #print(lambdaa)
    count = 0
    n = 10
    lambdaa = []
    while n <= 1000 : 
    
        for i in range(n):
            
            count = count + 1
            
            if count == (n-1):
                lambdaa.append(n)
                n = count + 11
                count = 0 
                
    x = "lambdaa" + "=" + str(z)
    #plt.plot(lambdaa , final_matrix , 'b', label = "train data")
    plt.plot(lambdaa , test_matrix , 'r', label = x )
    plt.xlabel("data size")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

# main function   
    
if __name__ == "__main__":
    train_100_10_data = np.genfromtxt("train-1000-100.csv", delimiter=',')
    train_label_100_10 = np.array(np.genfromtxt("trainR-1000-100.csv",delimiter = ","))
    
    matrix_data , matrix_label = split_data(train_100_10_data ,train_label_100_10 ) 
    #print(matrix_data[0])
    lambdaa = [5 , 25, 150]
    for z in lambdaa:
    
        final_matrix , matrix_for_w = training_set(matrix_data, matrix_label , z)
        len(matrix_for_w)
    
    
        test_100_10_data = np.genfromtxt("test-1000-100.csv", delimiter=',')
        test_100_10_label = np.genfromtxt("testR-1000-100.csv", delimiter=',')
        test_matrix = test_set(test_100_10_data,test_100_10_label,matrix_for_w)
    #print(len(test_matrix))
        plot_graphs(final_matrix , test_matrix , z)

  