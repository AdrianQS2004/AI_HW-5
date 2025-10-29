# Introduction to Artificial Intelligence
# Fashion MNIST Dataset
# Mnist_SoftmaxRegression1.py, code by Juan Carlos Rojas
# Adrian Quiros, Luis Baeza

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

#model = sklearn.linear_model.LogisticRegression(\
#    solver='sag', multi_class="ovr", max_iter = 50) 

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

# Compute the prediction accuracy against the training data
print("Against training set:")
pred_training = model.predict(train_data)
print("  Accuracy:   {:.3f}".format(sklearn.metrics.accuracy_score(train_labels, pred_training)))
print("  Precision:  {:.3f}".format(sklearn.metrics.precision_score(train_labels, pred_training, average='weighted')))
print("  Recall:     {:.3f}".format(sklearn.metrics.recall_score(train_labels, pred_training, average='weighted')))

#
# Explore coefficients
#
print("Min coef:", np.min(model.coef_))
print("Max coef:", np.max(model.coef_))
print("Coef mean:", np.mean(model.coef_))
print("Coef stddev: ", np.std(model.coef_))


# Plot a histogram of coefficient values
#"""
hist, bins = np.histogram(model.coef_, 500)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width)
plt.title("Coefficient values")
plt.show()
#"""

#"""
# Display the coefficients as an image, using a diverging colormap (red for negative, green for positive)
scale = 0.5
for n in range(num_classes):
    coef_img = model.coef_[n].reshape(28, 28)
    coef_img = np.clip(coef_img, -scale, scale)
    plt.figure()
    plt.imshow(coef_img, cmap="RdYlGn", vmin=-scale, vmax=scale)
    plt.title(f"Coefficients (clipped) for class {n}")
    plt.colorbar()
plt.show()