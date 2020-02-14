#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 04:21:53 2019

@author: vanshsmacpro
"""
import numpy as np
import sys
import timeit

# function where we perform cross validation and calculate MSE for all values of lambda
def cross_validation(train_data , train_label):
    min_mse = []
    for i in range(0,151):
    
        count = round(len(train_data)/10)
        
        train_d = []
        train_l = []
        split_data = []
        split_label = []
        train_matrix_data = []
        train_matrix_label = []
        test_matrix_data = []
        test_matrix_label = []
        final_mse = []
        y = count - (len(train_data) - 9*count) + 1
        
        for j in range((len(train_data)) + y):
            if j != count and j < len(train_data) : 
                train_d.append(train_data[j])
                train_l.append(train_label[j])
            elif j == count:
                split_data.append(train_d)
                split_label.append(train_l)
                count = count + round(len(train_data)/10)
                train_d = []
                train_l = []
                if(j < len(train_data)):
                    train_d.append(train_data[j])
                    train_l.append(train_label[j])
    #print(split_data[2])
    
        np.shape(split_data[0])
        np.shape(split_label[0])
        
        for k in range(10):
            for l in range(10):
                if(k != l):
                    train_matrix_data.append(split_data[l])
                    train_matrix_label.append(split_label[l])
                else:
                    test_matrix_label.append(split_label[l])
                    test_matrix_data.append(split_data[l])
                #np.shape(train_matrix_data[0])
        
            w = training_set(train_matrix_data , train_matrix_label,i)
            #print(w)
            MSE = test_set(test_matrix_label,test_matrix_data,w)
            #print(MSE)
            w =[]
            train_matrix_data =[]
            train_matrix_label=[]
            test_matrix_label=[]
            test_matrix_data =[]
            
            
            final_mse.append(MSE)
        #print(final_mse)
        min_mse.append(np.mean(final_mse))
    return min_mse
        
# function to calculate w for all lambda values
def training_set(train_matrix_data , train_matrix_label,i):
    t = []
    phi = []
    w = []
    phi_t = []
    B = []
    t_phi = []
    after_inverse = []
    inverse = []
    A = []
    I = []
    t = train_matrix_label[0]
    phi = train_matrix_data[0]
    len(train_matrix_data)
    np.shape(train_matrix_data)
    for j in range(1,len(train_matrix_data)):
        
        t = t + train_matrix_label[j] 
        phi = phi + train_matrix_data[j] 
        #np.shape(t)  
        #print(t)
        #print(t)
    len(t)
    len(phi)
    np.shape(t)
    phi = np.array(phi)
    np.shape(phi)
    t = np.array(t)
    np.shape(t)
    
    t = t.reshape(len(t), 1)
        #print(phi)
    phi_t = phi.T
    B = phi_t.dot(phi)
    I = np.identity(len(phi[0]), dtype = float)
    lambdaa = i
    A = I.dot(lambdaa)
    inverse = np.add(A,B)
    after_inverse = np.linalg.inv(inverse)
    
    t_phi = phi_t.dot(t)
    w = after_inverse.dot(t_phi)
    np.shape(w)
    
    return w

# function to calculate MSE for all lambda values

def test_set(test_matrix_label,test_matrix_data,w):
    t_test = []
    phi_test = []
    np.shape(test_matrix_data)
    t_test = test_matrix_label
    phi_test = test_matrix_data[0]
    len(test_matrix_data)
    np.shape(t_test)
    
    phi_test = np.array(phi_test)
    t_test = np.array(t_test)
    t_test = t_test[0]
    np.shape(t_test)
    len(phi_test)
    MSE = 0
        
    for i in range(len(phi_test)):
        
        MSE_phi = phi_test[i]
        t_i = t_test[i]
        target_variable = t_i
        MSE_phi = MSE_phi.reshape(len(MSE_phi),1)
        MSE_phi_t = MSE_phi.T
        
        pi_i = MSE_phi_t.dot(w)
        
        phi_w = pi_i
    
        phi_w = phi_w - target_variable
            
        phi_w = np.square(phi_w)
        #print(i)
    
        MSE = MSE + phi_w
   
    MSE = MSE/int(len(phi_test))
    return MSE
        #print(MSE)
        
        
# function to calculate w from train data 
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

# function to calculate test MSE
    
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
    
# main function
  
if __name__ == "__main__":
    start = timeit.default_timer()
    train_data = np.genfromtxt(sys.argv[1], delimiter=',')
    train_label = np.genfromtxt(sys.argv[2],delimiter = ",")
    
    
    test_100_10_data = np.genfromtxt(sys.argv[3], delimiter=',')
    test_100_10_label = np.genfromtxt(sys.argv[4], delimiter=',')
         
    len(train_data)
    average_mse = cross_validation(train_data ,train_label) 
    #print(average_mse)
    
   
    best = np.argmin(average_mse)
    print("best value of lambdaa is " , best)
    
    w  = training_set1(train_data, train_label , best)
    test_MSE = test_set1(test_100_10_data ,test_100_10_label, w )
    
    print("test MSE for the following lambda is ", test_MSE[0])
    
    
    
    
    end = timeit.default_timer()
    time = end - start
    print("running time is"  , time)