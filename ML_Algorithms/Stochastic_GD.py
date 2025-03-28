#create features matrix and calcualte target with deifned weights
# create random weights for predcition
# calculate pred_y using random weights
# shuffle the indexes and store them
# take each row using loop in shuffle indexes
# cal gradients, update gradients

import numpy as np
np.random.seed(42)
n = 300
x = np.random.randn(n,4) 
true_wts = np.array([2,3.5,3,5])
true_b = 2
true_y = x.dot(true_wts)+true_b+0.5*np.random.randn(n)
#print(true_y)
#intializing the random weitghs for pred target
wts = np.random.randn(4) #taking normal dist because of true_weights
b = np.random.rand()

alpha = 0.01
epochs = 200
indexes = np.arange(n)

for epoch in range(epochs+1):
    
    np.random.shuffle(indexes)

    for i in indexes:
        x_i = x[i]
        y_i = true_y[i]

        y_pred = x_i.dot(wts)+b
        error = (y_pred-y_i) #scalar value because of vector vector cal
        #print(error.shape)       

        grad_w = 2*(error*x_i)
        #print(grad_w.shape) 
        grad_b = 2*error

        wts -=  alpha*grad_w
        b -= alpha*grad_b

    if epoch%20 == 0:
        y_pred_all = x.dot(wts)+b
        loss = np.mean((true_y-y_pred_all)**2)
        print(f"loss at epoch_{epoch}:",loss)
    







