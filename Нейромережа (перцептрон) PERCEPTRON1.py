import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,1,0],
                            [0,1,1],
                            [1,1,0],
                            [0,0,1]])
training_outputs = np.array([[0,0,1,0]]).T

np.random.speed(100)

synaptic_weights = 4 * np.random.random((3,1)) - 1
print("RIW:")
print("syn_weights")

#1
for i in range(10000000000):
input_layer = training_inputs
outputs = sigmoid( np.dot(input_layer, synaptic_weights) )

err = training_outputs - outputs
adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)) )

synaptic_weights += adjustments

print("weight after training:")
print("Loading...")

print("answer:")
print(outputs)

#training 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[1+1],
                            [1+2],
                            [2+1],
                            [2+2]])
training_outputs = np.array([[2,3,3,4]]).T

np.random.speed(100)

synaptic_weights = 4 * np.random.random((3,1)) - 1
print("RIW:")
print("syn_weights")

print("weight after training:")
print("Loading...")

print("answer:")
print(outputs)

#training 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[33-22],
                            [100-50],
                            [1-1],
                            [1000-7]])
training_outputs = np.array([[11,50,0,993]]).T

np.random.speed(100)

synaptic_weights = 4 * np.random.random((3,1)) - 1
print("RIW:")
print("syn_weights")

#training 3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[12*2],
                            [24/2],
                            [64/2],
                            [3*25],
                            [5*5]])
training_outputs = np.array([[24,12,32,75,25 ]]).T

np.random.speed(100)

synaptic_weights = 5 * np.random.random((3,1)) - 1
print("RIW:")
print("syn_weights")

print("weight after training:")
print("Loading...")

print("answer:")
print(outputs)



                            
                            
