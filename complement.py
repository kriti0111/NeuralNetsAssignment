import numpy as np
import matplotlib.pyplot as plt


def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



#Input datasets for COMPLEMENT logic gate
inputs_set = np.array([[0,1],[1,1]]) 
expected_output_set = np.array([[0],[1]])

epochs = 30
lr = 0.5
inputLayerNeurons, outputLayerNeurons = 2,1

#Random weights and bias initialization(from -2 to 2)
output_weights = np.random.uniform(low=-2,high=2,size=(inputLayerNeurons,outputLayerNeurons))


print("Initial weights:")
print(*output_weights)

errors_set = []
weights_set = []
bias_set = []


#Training algorithm
for i in range(epochs):
	#Forward Propagation
	output_layer_activation = np.dot(inputs_set,output_weights)
	predicted_output = sigmoid(output_layer_activation)

	#Backpropagation
	error = expected_output_set - predicted_output
	errors_set.append(error)
	d_predicted_output = error * sigmoid_derivative(predicted_output)

	#Updating Weights and Biases
	output_weights += inputs_set.T.dot(error) * lr
	weights_set.append(output_weights.copy())


print("Final weights:")
print(*output_weights)

print("\nPredicted:")
print(*predicted_output)

errors_set = [sum(list(x.flatten())) for x in errors_set]
weights_0 = [list(x[0]) for x in weights_set]
weights_1 = [list(x[1]) for x in weights_set]

plt.plot(weights_0, label = "Weight1")
plt.plot(weights_1, label = "Bias")

plt.legend(bbox_to_anchor=(1,1), loc='upper left', borderaxespad=0.)
plt.show()