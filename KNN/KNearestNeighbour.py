import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#Splitting data to train and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier =  KNeighborsClassifier(n_neighbors= 5, metric='minkowski', p=2)
classifier.fit(X_train,y_train)

#Predicting Test Result
y_pred = classifier.predict(X_test)

#Making COnfusion Metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print cm













