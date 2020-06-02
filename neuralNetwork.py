
# numpy will allow us to use math functions and create matrices
# the scipy imports some important functions, like the sigmoid function

import numpy, scipy.special

class neuralNetwork:

    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):

        # This serves as the number of nodes that this NN will have.

        self.inNodes = input_layer
        self.hidNodes = hidden_layer
        self.outNodes = output_layer

        # This value represents the learning rate of the NN.

        self.learning = learning_rate

        # Creating the arrays of weights for the nodes' connections.
        # We limit the values of the weights depending on the number
        # of connections and to values from -0.5 to 0.5; otherwise
        # we risk saturating the sigmoid function.

        self.w_input_hidden = numpy.random.normal(0.0, pow(self.hidNodes, -0.5), (self.hidNodes, self.inNodes))
        self.w_hidden_output = numpy.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hidNodes))

        # Tweak to the activation function

        self.activation_func = lambda x: scipy.special.expit(x)

        pass

    def train(self, training_inputs, training_targets):

        # Converting the data to arrays

        # inputs = inputs list
        # targets = targets list

        inputs = numpy.array(training_inputs, ndmin=2).T
        targets = numpy.array(training_targets, ndmin=2).T

        # Signals into hidden layer
        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)

        # Same with output layer
        hidden_outputs = self.activation_func(hidden_inputs)

        # Final output calculation
        final_in = numpy.dot(self.w_hidden_output, hidden_outputs)
        final_out = self.activation_func(final_in)

        # Now, we need to redefine the weights linking each node/neuron
        # for this, we need to take the accumulated error of each layer.
        # We do this by back-propagation learning.

        # The out error = targets matrix - results matrix
        output_errors = targets - final_out

        # The hidden layer error = weight's matrix transposed * output_errors
        hidden_errors = numpy.dot(self.w_hidden_output.T, output_errors)

        # Updating the weights for the links between hidden and output
        self.w_hidden_output += self.learning * numpy.dot((output_errors * final_out * (1.0 - final_out)),numpy.transpose(hidden_outputs))

        # Updating the weighs for links between input and hidden
        self.w_input_hidden += self.learning * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass

    # Takes the inputs to the NN and returns the output. We use the linked
    # weights to moderate the signals, and also the sigmoid function to
    # activate or not the following node.

    def query(self, inputs):

        # Converting the inputs to a 2D array
        inputs = numpy.array(inputs, ndmin=2).T

        # Combines the inputs with the right link weights to produce a
        # matrix of combined moderated signals into each hidden layer node.
        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)

        # The signal from the input nodes to the output nodes - we use the
        # previous moderated signals
        hidden_outputs = self.activation_func(hidden_inputs)

        # Calculate signals into output layer
        final_in = numpy.dot(self.w_hidden_output, hidden_outputs)

        # Signals from the final output
        final_out = self.activation_func(final_in)

        return final_out