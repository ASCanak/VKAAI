import time
import numpy as np

class activation_functions:
    def sigmoid(x):
        """Returns sigmoid of x."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        """Returns sigmoid derivative of x."""
        return activation_functions.sigmoid(x) * (1 - activation_functions.sigmoid(x))

class progress_bar:
    def show_progress(current, total, bar_length=50):
        """Displays a dynamic progress bar."""
        percent = 100 * current / total
        arrow   = '-' * int(percent / 100 * bar_length) + '>'
        spaces  = ' ' * (bar_length - len(arrow))
        print("\rProgress: [{}] {:.2f}%".format(arrow + spaces, percent), end='', flush=True)

class data_loader:
    def load_dataset(filename):
        """Load data and outputs from file."""
        data    = np.genfromtxt(filename, delimiter=",", usecols=[0, 1, 2, 3])
        outputs = np.genfromtxt(filename, delimiter=",", usecols=[4], dtype=str)
        return data, outputs
    
    def serialize(outputs):
        """Convert types to a list of length of types where correct type is 1 and incorrect is 0."""
        return (np.tile(outputs, (3, 1)).T == np.unique(outputs)).astype(int)

    def normalize(data):
        """Normalize data."""
        max_values = np.amax(data, axis=0)
        min_values = np.amin(data, axis=0)
        return (data - min_values) / (max_values - min_values)

    def shuffle(data, outputs):
        """Shuffle data and outputs."""
        shuffle_index = np.arange(data.shape[0])
        np.random.shuffle(shuffle_index)
        return data[shuffle_index], outputs[shuffle_index]

    def split_by_percentage(array, percentage):
        """Split an array into two parts based on a given percentage."""
        return array[:int(array.shape[0] * percentage)], array[int(array.shape[0] * percentage):]

class neuron:
    def __init__(self, previous_layer=[]):
        """Set parameters for Neuron (use default params for input layer)."""
        self.previous_layer   = previous_layer
        self.weights          = np.random.uniform(-1, 1, len(self.previous_layer))
        self.bias             = np.random.uniform(-1, 1)
        self.error            = 0
        self.output           = 0
        self.activated_output = 0

    def forward_propagation(self):
        """Propagate the input values of the previous layer with the corresponding weights to calculate its output."""
        self.output = np.dot(np.array([n.activated_output for n in self.previous_layer]), self.weights) + self.bias
        self.activated_output = activation_functions.sigmoid(self.output)
        return self.activated_output

    def update(self, learn_rate):
        """Update weights and bias."""
        self.weights += learn_rate * self.error * np.array([n.activated_output for n in self.previous_layer])
        self.bias    += learn_rate * self.error

class neural_network:
    def __init__(self, input_layer_size, hidden_layer_size, output_layers_size, learn_rate=0.01):
        """Initializes all layers to train and classify data with."""
        self.learn_rate = learn_rate
        
        if hidden_layer_size > 0:
            self.input_layer  = [neuron(                 ) for _ in range(input_layer_size  )]
            self.hidden_layer = [neuron(self.input_layer ) for _ in range(hidden_layer_size )]
            self.output_layer = [neuron(self.hidden_layer) for _ in range(output_layers_size)]
        else: 
            self.input_layer  = [neuron(                ) for _ in range(input_layer_size  )]
            self.hidden_layer = None
            self.output_layer = [neuron(self.input_layer) for _ in range(output_layers_size)]

    def predict(self, input_data):
        """Predicts classification of input_data which is a list the length of self.output_layers."""
        for input_neuron, data in zip(self.input_layer, input_data):
            input_neuron.activated_output = data

        if self.hidden_layer is not None:
            for hidden_neuron in self.hidden_layer:
                hidden_neuron.forward_propagation()

        predicted_outputs = [output_neuron.forward_propagation() for output_neuron in self.output_layer]
        return predicted_outputs

    def back_propagation(self, desired_output):
        """Back propagates the error of the hidden and output layers."""
        for desired_output_index, output_neuron in enumerate(self.output_layer):
            output_neuron.error = activation_functions.sigmoid_derivative(output_neuron.output) * (desired_output[desired_output_index] - output_neuron.activated_output)
        
        if self.hidden_layer is not None:
            for weight_index, hidden_neuron in enumerate(self.hidden_layer):
                error_sum = sum(output_neuron.error * output_neuron.weights[weight_index] for output_neuron in self.output_layer)
                hidden_neuron.error = activation_functions.sigmoid_derivative(hidden_neuron.output) * error_sum

    def update_weights(self):
        """Updates weights and bias of hidden and output layers."""
        if self.hidden_layer is not None:
            for hidden_neuron in self.hidden_layer:
                hidden_neuron.update(self.learn_rate)

        for output_neuron in self.output_layer:
            output_neuron.update(self.learn_rate)

    def train(self, input_data, desired_output, epochs):
        """Trains layers with input_data and desired_output to classify similar data correctly for a given number of epochs."""
        for epoch_index in range(epochs):
            progress_bar.show_progress(epoch_index, epochs-1)
            for data_in, data_out in zip(input_data, desired_output):
                self.predict(data_in)
                self.back_propagation(data_out)
                self.update_weights()

if __name__ == "__main__":
    np.random.seed(0)

    # Load and preprocess dataset
    data, outputs      = data_loader.load_dataset("Dataset/iris.data")
    normalized_data    = data_loader.normalize(data)
    serialized_outputs = data_loader.serialize(outputs)

    # Split data into training and testing sets
    train_data, test_data     = data_loader.split_by_percentage(normalized_data, 0.85)
    train_output, test_output = data_loader.split_by_percentage(serialized_outputs, 0.85)

    # Train the neural network
    time1 = time.perf_counter()
    nn = neural_network(input_layer_size=4, hidden_layer_size=8, output_layers_size=3, learn_rate=0.1)
    nn.train(input_data=train_data, desired_output=train_output, epochs=100)
    print(f"\nTraining took {(time.perf_counter() - time1):0.2f} seconds")

    # Evaluate accuracy on the test_data
    correct_predictions = np.all(np.rint([nn.predict(data_point) for data_point in test_data]) == test_output, axis=1)
    accuracy = np.count_nonzero(correct_predictions) / len(test_data) * 100
    print(f"Neural Network accuracy on test data: {accuracy:0.1f}%")

    # Evaluate accuracy on the entire dataset
    correct_predictions = np.all(np.rint([nn.predict(data_point) for data_point in normalized_data]) == serialized_outputs, axis=1)
    accuracy = np.count_nonzero(correct_predictions) / len(normalized_data) * 100
    print(f"Neural Network accuracy on entire dataset: {accuracy:0.1f}%")

    # Make a prediction for a specific index
    index = 50
    print(f"\nIndex {index} is {outputs[index]}")
    prediction = nn.predict(normalized_data[index])
    print(f"{prediction[0] * 100 :0.4f}% Iris-Setosa")
    print(f"{prediction[1] * 100 :0.4f}% Iris-Versicolor")
    print(f"{prediction[2] * 100 :0.4f}% Iris-Virginica\n")