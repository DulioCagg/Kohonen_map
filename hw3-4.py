from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt
import time


def classification_graph(file, data_set):
    """
    Prints and saves the winner neurons of the passed data set
    """
    # Initialize a graph of dimentions 8 by 8
    plt.figure(figsize=(8, 8))
    plt.pcolor(som.distance_map().T, cmap='bone_r')

    # Extracts the names of the flowers provided in the dataset
    f = "data_sets/{}.csv".format(file)
    target = np.genfromtxt(f, delimiter=',', usecols=(4), dtype=str)
    target_num = np.zeros(len(target), dtype=int)
    target_num[target == "setosa"] = 0  # Green
    target_num[target == "versicolor"] = 1  # Red
    target_num[target == "virginica"] = 2  # Blue

    markers = ["o", "s", "D"]
    colors = ["green", "red", "blue"]

    # It marks the winner neuron in the map with the marker and color corresponding to that flower
    for index, value in enumerate(data_set):
        w = som.winner(value)
        plt.plot(w[0]+0.5, w[1]+0.5, markers[target_num[index]], markerfacecolor="None", markeredgecolor=colors[target_num[index]], markersize=12, markeredgewidth=2)

    plt.title("Classification for {}".format(file))
    plt.axis([0, 8, 0, 8])
    save_dir = "results/{}.png".format(file)
    plt.savefig(save_dir)
    plt.show()


def frequency_graph(data_set):
    """
    Prints and saves the frequency map, the darker the color the more activations got the neuron
    """
    # Initialize a graph of dimentions 8 by 8
    plt.figure(figsize=(8, 8))
    frequencies = som.activation_response(data_set)
    plt.pcolor(frequencies.T, cmap="Blues")
    plt.colorbar()
    plt.title("Map Frequency")
    plt.savefig("results/map_frequency.png")
    plt.show()


def error_graph(data_set):
    """
    Prints and saves how the error changes over the training epochs
    """
    max_iter = 10000
    q_error = []
    t_error = []
    iter_x = []
    for i in range(max_iter):
        percent = 100*(i+1)/max_iter
        rand_i = np.random.randint(len(data_set))
        som.update(data_set[rand_i], som.winner(data_set[rand_i]), i, max_iter)
        if (i+1) % 100 == 0:
            q_error.append(som.quantization_error(data_set))
            t_error.append(som.topographic_error(data_set))
            iter_x.append(i)

    plt.plot(iter_x, q_error)
    plt.title("Error over time")
    plt.ylabel('Error')
    plt.xlabel(("Number of epochs"))
    plt.savefig("results/error.png")
    plt.show()


def get_dataset(file):
    """
    Returns the normalized data set of the speficied file
    """
    data = np.genfromtxt(file, delimiter=",", usecols=(0, 1, 2, 3))
    return np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)


train_dataset = get_dataset("data_sets/train.csv")
test_dataset = get_dataset("data_sets/test.csv")
validation_dataset = get_dataset("data_sets/validation.csv")

# Creates SOM of 8x8 dimentions
som = MiniSom(8, 8, 4, sigma=1.3, learning_rate=0.5)
# Initializes the weights with info of the dataset
som.pca_weights_init(train_dataset)

print("Training started")
start = time.perf_counter()
som.train_batch(train_dataset, 10000)
end = time.perf_counter()
print("Training took {} seconds!".format(end - start))

# Generation of graphs
classification_graph("test", test_dataset)
classification_graph("validation", validation_dataset)
classification_graph("train", train_dataset)
frequency_graph(train_dataset)
error_graph(train_dataset)
