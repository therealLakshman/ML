#create features matrix and create  target vector
#take a random weights and random interceptor
#create a formula and predict target variable
# calcualte loss
#declare epocs  and learning rate alpha
# run epoch loop
# create gradient formula for weights and interceptor 
# calculate gradient descent 
# for defined no. of epoch cal error 
import numpy as np
np.random.seed(42)
n = 100
x = np.random.randn(n,3)
true_wts = np.array([2,5,7])
true_b = 2
# print("real x,y shapes",x.shape,true_wts.shape)
true_y = x.dot(true_wts)+true_b+0.25*np.random.randn(n) #adding random noise
# print(true_y)
wts = np.random.rand(3)# randomly intialies the weitghs unifrom dist
b = np.random.rand()
#errors = y_pred-true_y
#print(errors)
x_transposed = x.T #helps us prevent making transpose of x matrix for evry epoch during wts gradient calculation
epochs = 1000
alpha = 0.1
#for early stopping
prev_loss = None # intilaizing with none will stop breaking loop if first epoch loss is very less
tolerance = 1e-7
for epoch in range(epochs+1):
    # wnew = wold - eta*1/m((y_pred-y_true)*x)
    # bnew = bold - eta*1/m(errors) 
    y_pred = x.dot(wts)+b
    errors = y_pred-true_y
    gradient_w = 2/n*(x_transposed.dot(errors))
    gradient_b = 2/n*(np.sum(errors))
    step_size_w = alpha*gradient_w
    step_size_b = alpha*gradient_b
    
    #updating the weights 
    wts -= step_size_w
    b -= step_size_b

    loss = np.mean(errors**2)
    if epoch%100 ==0:
        print(f"loss at epoch_{epoch}:",loss)
   
    if prev_loss is not None and abs(prev_loss-loss) < tolerance:
        print(f"converged at epoch{epoch}")
        if epoch%100 !=0:
            print(f"loss at epoch_{epoch}:",loss)
        break
    prev_loss = loss
    





