# Introduction to Artificial Intelligence
# Fashion MNIST Dataset
# Mnist_Explore, code by Juan Carlos Rojas
# Adrian Quiros, Luis Baeza

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the training and test data from the Pickle file
with open("fashion_mnist_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale pixels to range 0-1.0
maxval = train_data.max()
train_data = train_data / maxval
test_data = test_data / maxval

# Print some information about the training dataset
print("Training dataset size: ", train_data.shape)
print("Class histogram: ")
print(np.histogram(train_labels, 10)[0])

# Print some information about the test dataset
print("Test dataset size: ", test_data.shape)
print("Class histogram: ")
print(np.histogram(test_labels, 10)[0])

# Plot a histogram of pixel values
#"""
hist, bins = np.histogram(train_data, 50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Train dataset pixel values")
plt.show()
#"""

# Plot a few images
"""
for idx in range(5):
  image = train_data[idx].reshape(28,28)
  plt.figure()
  plt.imshow(image, cmap="gray_r")
  plt.title("Label: "+str(train_labels[idx]))
plt.show()
"""



# Compute the centroid image from training (the mean pixel value)
centroid_img = np.mean(train_data, 0)
plt.figure()
plt.imshow(centroid_img.reshape(28,28), cmap="gray_r")
plt.title("Centroid Training")
plt.show()

# Compute the centroid image from test (the mean pixel value)
centroid_img_test = np.mean(test_data, 0)
plt.figure()
plt.imshow(centroid_img_test.reshape(28,28), cmap="gray_r")
plt.title("Centroid Test")
plt.show()

#"""
# Compute an average image per class
class_list = np.unique(train_labels)
num_classes = len(class_list)
for classidx in range(num_classes):
  
  # Create an image of average pixels for this class
  mask = train_labels==classidx
  train_data_this_class = np.compress(mask, train_data, axis=0)

  mean_img_in_class = np.mean(train_data_this_class, 0)
  plt.figure()
  plt.imshow(mean_img_in_class.reshape(28,28), cmap="gray_r")
  plt.title("Centroid for Class "+str(classidx))

  plt.show()
#"""


