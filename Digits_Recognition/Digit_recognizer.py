import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt


# importing data
digits = datasets.load_digits()


# Preparing data
images = digits.images
labels = digits.target

n_samples = len(images)
images = images.reshape((n_samples, -1))


n_train = n_samples//2
n_test = n_samples - n_train

# Splitting Data into Train and Test
X_train = images[:n_train]
X_test = images[n_train:]

Y_train = labels[:n_train]
Y_test = labels[n_train:]


#Creating the model

svm_model = svm.SVC(gamma=0.001)

svm_model.fit(X_train,Y_train)


#Accuracy
predicted = svm_model.predict(X_test)

print(classification_report(Y_test, predicted ))

confusion_matrix = confusion_matrix(Y_test, predicted)
#disp = metrics.plot_confusion_matrix(svm, X_test, Y_test)
#disp.figure_.suptitle("Confusion Matrix")
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
disp.plot()
plt.show()

print("The accuracy of the SVM algorithm is : ")
print(svm_model.score(X_test,Y_test))
