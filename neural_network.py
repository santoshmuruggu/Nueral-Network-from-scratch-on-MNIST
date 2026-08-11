import numpy as np 

# Input to First hidden layer.
x = np.array([4, -2])
w = np.array([
             [3,1,0],
             [1,1,5]
             ])
b = np.array([0,2,1])

output = np.dot(x,w) + b
#print(output) 

def relu(z) : 
    return np.maximum(0,z)

activated = relu(output)
print("input layer after activation: ")
print(activated) 

# hidden layer to output2 layer. 
w2 = np.array([
    [1,0], 
    [2,1], 
    [0,3]
])
b2 = np.array([1,0])

output2 = np.dot(activated, w2) + b2
print("output2: ")
print(output2)

# moving to the softmax. 
import numpy as np
scores = np.array([19,4])
exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores)
print("probability:")
print(probs)

#print(np.exp(-9))
#print(np.exp(0))
#print(np.exp(19))


## Building the loss fuction. 
probs = np.array([0.9999997, 0.0000003])
true_class = 1

loss = -np.log(probs[true_class])
print("loss: ")
print(loss)
