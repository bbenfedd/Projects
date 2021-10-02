import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


#you need to be placed in same directory where winequality-red.csv is placed
#you can uss the method chdir() from the module os
#os.chdir()


data = pd.read_csv("winequality-red.csv")

#Attribute vector
X = data.iloc[:,:11]


#Convert a pandas Data Frame to a numpy array
X = pd.DataFrame.to_numpy(X)

#Quality vector
Y =data.iloc[:,11]
#Convert a pandas Data Frame to a numpy array
Y = pd.DataFrame.to_numpy(Y)



#create and fit the model using Scikit-Learn
model = LinearRegression()
model.fit(X,Y)


beta = np.array([model.coef_ ,model.intercept_])









