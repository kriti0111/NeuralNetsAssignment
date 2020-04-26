import numpy as np
import matplotlib.pyplot as plt

#Input datasets 
inputs_set = np.array([[1,0], [1 ,0.8] , [1, 1.6], [1, 3] , [1, 4] , [1 ,5]]) 
output_set = np.array([[0.5],[1],[4],[5],[6],[9]])

inputs = np.array([[0],[0.8],[1.6],[3],[4],[5]])

epochs = 100
lr = 0.01
inputLayerNeurons, outputLayerNeurons = 2,1

#Random weights and bias initialization(from -2 to 2)
weights = np.random.random(size=(inputLayerNeurons,outputLayerNeurons))
print("initial weights ")
print(weights)
weights_set = []
total_error = []
calc_error = [0]
for i in range(epochs):

	#Backpropagation
    error = output_set - np.dot(inputs_set,weights)
    calc_error =  (error**2)/2
    total_error.append(calc_error.copy())

    #Updating Weights and Biases
    weights += inputs_set.T.dot(error) * lr
    weights_set.append(weights.copy())


print("Final weights:")
print(*weights)

total_error = [sum(list(x.flatten())) for x in total_error]
weights_0 = [list(x[0]) for x in weights_set]
weights_1 = [list(x[1]) for x in weights_set]


#plt.plot(total_error)


f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax2 = f2.add_subplot(111)


ax1.set_title('Weight Trajectories vs epochs(number of iterations)')
ax1.set_xlabel('Number of iterations')
ax1.plot(weights_0, label = "bias")
ax1.plot(weights_1, label = "Weight")
ax1.legend(bbox_to_anchor=(1,1), loc='upper left', borderaxespad=0.)

x = np.linspace(0,10,100)
y = (weights_1[99])*x+ (weights_0[99])
ax2.set_title('LMS fitting result')
ax2.plot(x, y, '-r')
ax2.scatter(inputs, output_set)
ax2.grid()

plt.show()
