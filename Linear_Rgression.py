# %%
import numpy as np
import matplotlib.pyplot as plt
#from utils import load_data
import copy
import math
# %% Loading The DAtaset
data = np.loadtxt("data/ex1data1.txt", delimiter= ',')
x_train = data[:,0]
y_train = data[:,1]

# %% Length of the training dataset
print ('Length of training data (m)', len(x_train))
# Plot the dataset
plt.scatter(x_train, y_train)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# %%  model function = 𝑓𝑤,𝑏(𝑥)=𝑤𝑥+𝑏
# to find the best w & b fit to the dataset
# we can implement the cost function that depends on w & b
# 𝐽(𝑤,𝑏)=1/2𝑚∑(𝑖=0𝑚−1)(𝑓𝑤,𝑏(𝑥^(𝑖))−𝑦^(𝑖))^2


def compute_cost(x, y, w, b):
    m = x.shape [0]     #number of training examples
    total_cost = 0   #declare the cost variable

    for i in range (m):  #for all the training examples 
        f_wb = w*x[i] + b     #calculate the predicted output for all training ex
        cost = (f_wb - y[i])**2     #cost = predicted - actual
        total_cost += cost          
    total_cost = total_cost / 2*m   #J_wb = 
    return total_cost
# %%
# we can use Gradient Decent algorithm on the cost function 
# to find the best w & b.
#repeat until convergence:{𝑏:=𝑏−𝛼∂𝐽(𝑤,𝑏)/∂𝑏𝑤:=𝑤−𝛼∂𝐽(𝑤,𝑏)/∂𝑤}

def compute_gradient(x, y, w, b):
    m = len(x)

    dj_dw = 0
    dj_db = 0 #declare variables for the G. D.

    for i in range (m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])             #when simplify the derivatives
    #∂𝐽(𝑤,𝑏)/∂𝑏=1/𝑚∑(𝑖=0 𝑚−1) ∂𝐽(𝑤,𝑏)(𝑖)/∂𝑏
    #∂𝐽(𝑤,𝑏)/∂𝑤=1/𝑚∑(𝑖=0 𝑚−1) ∂𝐽(𝑤,𝑏)(𝑖)/∂𝑤
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db
# %%Implementing some unit tests


# %%
#initializing parameters
w = 0
b = 0

#initializing Hyperparameters
alpha = 0.01    #Learning RAte
num_itr = 1500

#Gradient DEcent iterations
for i in range(num_itr):
    dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)
    #update paras
    w = w - alpha * dj_dw
    b = b - alpha * dj_db

    #prininting cost
    if i % 100 == 0:
        cost = compute_cost(x_train, y_train, w, b)
        print (f'Iteration {i}   Cost : {cost}')

print(f'weight= {w}, bias= {b}')
# %%
