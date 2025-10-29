# Introduction to Artificial Intelligence
# Fashion MNIST Dataset
# Mnist_SoftmaxRegression1.py, code by Juan Carlos Rojas
# Adrian Quiros, Luis Baeza

# The main difference to the referenced code is the 
# added Part 5 which displays wrong images.
# that specific part 5 code also comes from another program
# given to us by the professor. 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.linear_model

#
# Load and prepare input data
#

# Load the training and test data from the Pickle file
with open("fashion_mnist_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale pixels to range 0-1.0
maxval = train_data.max()
train_data = train_data / maxval
test_data = test_data / maxval

#
# Train classifier
#

# Train a Softmax Regression classifier
# Use stochastic approach to save time

model = sklearn.linear_model.LogisticRegression(\
    solver='sag', max_iter = 50) 

print("Training model")
model.fit(train_data, train_labels)

# Make the class predictions
pred = model.predict(test_data)

#
# Metrics
#

# Confusion matrix
cmatrix = sklearn.metrics.confusion_matrix(test_labels, pred)
print("Confusion Matrix:")
print(cmatrix)

# Accuracy, precision & recall
print("Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(test_labels, pred)))
print("Precision:  {:.3f}".format(sklearn.metrics.precision_score(test_labels, pred, average='weighted')))
print("Recall:     {:.3f}".format(sklearn.metrics.recall_score(test_labels, pred, average='weighted')))

# Per-Class Precision & Recall
precision = sklearn.metrics.precision_score(test_labels, pred, average=None)
recall = sklearn.metrics.recall_score(test_labels, pred, average=None)
num_classes = len(np.unique(train_labels))
for n in range(num_classes):
    print("  Class {}: Precision: {:.3f} Recall: {:.3f}".format(n, precision[n], recall[n]))


# Part 5, displaying wrong images
num_displayed = 0
x = 0
while (num_displayed < 10):
    x += 1

    # Skip correctly predicted 
    if (pred[x] == test_labels[x]):
        continue

    num_displayed += 1

    # Display the images
    image = test_data[x].reshape(28,28)
    plt.figure()
    plt.imshow(image, cmap="gray_r")
    plt.title("Predicted: "+str(pred[x])+" Correct: "+str(test_labels[x]))
    plt.show()


