# create size ,features matrix and defined weights
# using the weights and features generate target variables
# create random weigths and bias
# declare epoch and minibatch size
# run the epoch loop
# create indexes and shuffle them as it is very import to give 
# run the minibatch loop

import numpy as np
np.random.seed(42)
n = 300
x = np.random.randn(n,4)
true_wts = np.array([2,3.5,3,5])
true_b = 2
true_y = x.dot(true_wts)+true_b+0.5*np.random.randn(n)
#print(true_y)

#creating random weights
wts = np.random.randn(4)
b = np.random.rand()

epochs = 200
alpha = 0.01
batch_size = 32
indexes = np.arange(n)

for epoch in range(epochs+1):
    np.random.shuffle(indexes)
    for start in range(0,n,batch_size):
       batch_indices = indexes[start:start+batch_size]
       #print("size of batch_indexes",len(batch_indices))
       x_shuffled = x[batch_indices]
       #print("x_shuffled",x_shuffled.shape)
       y_shuffled = true_y[batch_indices]

       y_pred = x_shuffled.dot(wts)+b
       #print(y_pred.shape)
       error = y_shuffled-y_pred #actual_y-predicted_y so we will get negative sign before gradient

       gradient_w = -(2/batch_size)*(x_shuffled.T.dot(error)) #nxm-mX1 -> nX1 -> to flatten(n,)
       #print("grad_w",gradient_w.shape)
       gradient_b = -2*np.mean(error)

       wts -= alpha*gradient_w
       b -= alpha*gradient_b

    if epoch%20 ==0:
        y_pred_all = x.dot(wts)+b
        loss = np.mean((y_pred_all-true_y)**2)
        print(f"loss at epoch_{epoch}:",loss)



