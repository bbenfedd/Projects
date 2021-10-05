import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt


# importing data
digits = datasets.load_digits()


# Preparing data
images = digits.images
labels = digits.target

n_samples = len(images)
images = images.reshape((n_samples, -1))


n_train = n_samples//2
n_test = n_samples - n_fit

# Splitting Data into Train and Test
X_train = images[:n_train]
X_test = images[n_train:]

Y_train = labels[:n_train]
Y_test = labels[n_train:]


#Creating the model

svm_model = svm.SVC(gamma=0.001)

svm_model.fit(X_train,Y_train)


predicted = svm_model.predict(X_test)
