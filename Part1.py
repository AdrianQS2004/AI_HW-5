# Introduction to Artificial Intelligence
# Fashion MNIST Dataset
# Preprocessing of data, code by Juan Carlos Rojas
# Adrian Quiros, Luis Baeza

# This code is really similar to the referenced code, few changes done 
# The main difference is that this code ends by writing the data in a pickle file. 

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Modified by JCR to work on NumPy arrays
def get_mnist_data_and_labels(images_filename, labels_filename):
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")
    try:
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_image_values = num_of_rows * num_of_colums
        print("Number of images: ", num_of_items)
        print("Size of images: ", num_of_rows, "by", num_of_colums)
        labels_file.read(8)
        data = [[None for x in range(num_of_image_values)]
                for y in range(num_of_items)]
        labels = []
        for item in range(num_of_items):
            if (item % 1000) == 0:      
                  print("Current image number: %7d" % item)
            for value in range(num_of_image_values):
                data[item][value] = int.from_bytes(images_file.read(1),
                                                   byteorder="big")
            labels.append(int.from_bytes(labels_file.read(1), byteorder="big"))
        # Convert to NumPy arrays
        data = np.array(data).astype(np.uint8)
        labels = np.array(labels).astype(np.uint8)
        return data, labels
    finally:
        images_file.close()
        labels_file.close()
        print("Complete")

# Read the train data and labels files
print("Reading training dataset")
train_data, train_labels = get_mnist_data_and_labels("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
train_size = train_data.shape[0]

print("Reading test dataset")
test_data, test_labels = get_mnist_data_and_labels("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
test_size = test_data.shape[0]

# Archive the data into a pickle file
with open("fasion_mnist_dataset.pickle", 'wb') as f:
      pickle.dump([train_data, train_labels, test_data, test_labels], \
                  f, pickle.HIGHEST_PROTOCOL)

