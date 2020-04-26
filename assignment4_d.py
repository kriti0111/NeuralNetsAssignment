import numpy as np
import matplotlib.pyplot as plt

#Input datasets 
inputs_set = np.array([[1,0], [1 ,0.8] , [1, 1.6], [1, 3] , [1, 4] , [1 ,5]]) 
output_set = np.array([[0.5],[1],[4],[5],[6],[9]])

inputs = np.array([[0],[0.8],[1.6],[3],[4],[5]])

epochs = 100
lr_set = [0.01,0.001, 0.02, 0.0001]#[0.05]
inputLayerNeurons, outputLayerNeurons = 2,1
counter = 0

while counter < 4:
    weights = np.random.random(size=(inputLayerNeurons,outputLayerNeurons))
    print("initial weights ")
    print(weights)
    total_error = []
    calc_error = [0]
    lr = lr_set[counter]
    for i in range(epochs):

	    #Backpropagation
        error = output_set - np.dot(inputs_set,weights)
        calc_error =  (error**2)/2
        total_error.append(calc_error.copy())

        #Updating Weights and Biases
        weights += inputs_set.T.dot(error) * lr


    
    counter += 1
    total_error = [sum(list(x.flatten())) for x in total_error]
   
    plt.plot(total_error, label = str(lr))
plt.legend(bbox_to_anchor=(1,1), loc='upper left', borderaxespad=0.)    
plt.show()


