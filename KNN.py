import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Importing the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 8].values

# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle=False) # 67% training and 30% test
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=2)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
print("predicted",y_pred)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy for KNN :",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


