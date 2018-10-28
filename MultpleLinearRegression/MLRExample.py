import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Encoding Categorical Data
#Encoding Independent Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:, 3])
onehotencoder  =  OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X=X[:, 1:]

#Splitting data to train and test dataset

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)
#modelscore = regressor.score(X, y)

#Predicting the Test Set Results
y_pred=regressor.predict(X_test)

#FInd Accuracy Score
from sklearn.metrics import accuracy_score

#accuracyscore = accuracy_score(y_test, y_pred, normalize=False)






