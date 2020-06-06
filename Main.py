import numpy, matplotlib.pyplot as plt
from neuralNetwork import neuralNetwork

# nodes and learning rate
input_layer = 784
hidden_layer = 100 # Optimal is 200, but it takes considerably longer to train and the % difference is about 1%
output_layer = 10
learning_rate = 0.1

# Instance of neural network
nn = neuralNetwork(input_layer, hidden_layer, output_layer, learning_rate)

# Loading training dataset - this code is for the small sample
# training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
# training_data_list = training_data_file.readlines()
# training_data_file.close()

# Loading training dataset - this code is for the big sample. You need to unzip the full data sets
training_data_file = open("mnist_dataset\mnist_full_dataset\mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# TRAINING - Epochs are the naming standard for iterative trainings

#   We set the epoch = 2; which means that we will perform two training
#   sessions with the same training list of values, which means that the
#   training time will double.

epochs = 5

for epoch in range(epochs):

    # Neural Network training - iteration through all data sets
    for entry in training_data_list:

        all_values = entry.split(',')

        # For the NN to work, we need the values to be in the range from 0.01 to 0.99

        scaled_inputs = (numpy.asfarray(all_values[1:]) / 255.0*0.99) + 0.01

        # The targeted values: they start at the beginning of each data set (for example,
        # the first number in the data set is 5, therefore the values afterward correspond
        # to the number 5.
        targets = numpy.zeros(output_layer) + 0.01

        # the target value we mentioned before. it will allocate the number 0.99 in the array
        targets[int(all_values[0])] = 0.99
        nn.train(scaled_inputs, targets)

pass

# Neural Network testing - same as before, but we expect the correct result as output
# from the NN. This time, we load the test data for the test process

# This code is for the small data set sample
# test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
# test_data_list = test_data_file.readlines()
# test_data_file.close()

# This code is for the big data set sample. You need to unzip the full data sets
test_data_file = open("mnist_dataset\mnist_full_dataset\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Score card gives points to check how well the NN has performed
score_card = []

# Iterate through each test

for entry in test_data_list:

    all_values = entry.split(',')
    correct_answer = int(all_values[0])
    print("The expected value is " + all_values[0])

    # scaling the test inputs and querying them
    test_outputs = nn.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)

    # Chosen result of the NN in the test outputs
    answer = numpy.argmax(test_outputs)
    print("Neural Network answered:", answer)

    # Checking whether the result is correct or not
    if answer == correct_answer:
        # adding one point to the score card
        score_card.append(1)
    else:
        # if the NN is mistaken, we add a 0 to the score card
        score_card.append(0)
    pass

    # This code displays the number in order to see how does it look like.
    # NOTE: USE IT JUST WITH THE SMALL SAMPLES
    # image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
    # plt.imshow(image_array, cmap='Greys', interpolation='None')
    # plt.show()

# Calculate the accuracy of the Neural Network
accuracy = (numpy.sum(score_card) / len(score_card)) * 100.0
print("The Neural Network accuracy is:", accuracy, "%")