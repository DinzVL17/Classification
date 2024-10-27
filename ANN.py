'''
Title: Implementing the ANN
Author: Dinuri Vishara
Date: 07/02/2023
'''
# load the Iris dataset into a Pandas DataFrame
import pandas as pd
iris = pd.read_csv("iris.csv")
# print(iris)

# split the measurement data and class labels
X = iris.iloc[:,[0,1,2,3]]
# print(X)
y = iris.iloc[:,5]
# print(y)

# species category values of the Y variable
label = pd.unique(y)
# print(label)
# replace the category values with integer numbers
from sklearn import preprocessing
ylabel= preprocessing.LabelEncoder()
ylabel.fit(label)
yvalue= ylabel.transform(y)
yvalue = yvalue.ravel()
# print(yvalue)

# split the Iris dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,yvalue,test_size=0.20)
# print(X_train)

# standardize the training and testing data of the input variable
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_trainS = scaler.transform(X_train)
scaler.fit(X_test)
X_testS = scaler.transform(X_test)
# print(X_testS)

# train ANN with 3 hidden layers
from sklearn.neural_network import MLPClassifier
mpl = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000).fit(X_trainS,y_train)
print(mpl)
# predict the species labels of the test data
predict = mpl.predict(X_testS)
print(predict)


# confusion_matrix and classification_report
from sklearn.metrics import confusion_matrix,classification_report
# print(confusion_matrix(y_true=y_test,y_pred=predict))
# print(classification_report(y_true=y_test,y_pred=predict))

import numpy as np
testplant = np.array([[5.9,3.0,7.0,5.0],[4.6,3.0,1.5,0.2],[6.2, 3.0,4.1,1.2]])
# print(testplant)

# preprocess testdata
scaler.fit(testplant)
testplantSTD = scaler.transform(testplant)
# predict the species of the 3 plants
pred = mpl.predict(testplantSTD)
predlabel = ylabel.inverse_transform(pred)
print("predicted label:\n",pred)
print("Species:\n",predlabel)

# train ANN with 2 neurons in each hidden layer
mpl2 = MLPClassifier(hidden_layer_sizes=(2,2,2), max_iter=1000).fit(X_trainS,y_train)
predict2 = mpl.predict(X_testS)
print(predict2)
# confusion_matrix and classification_report
print(confusion_matrix(y_true=y_test,y_pred=predict2))
print(classification_report(y_true=y_test,y_pred=predict2))



















