import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split



 
loadeddata = pd.read_csv('data.txt', names = ["Height", "Weight", "Gender"])

X = loadeddata.drop('Gender', axis=1)
y = loadeddata['Gender']

# 75% of data for testing, using smooth/soft logistic sigmoid function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, train_size=0.25, random_state=42)


scaler1 = StandardScaler()
scaler1.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler1.transform(X_train)
X_test = scaler1.transform(X_test)

mlp1 = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000, activation='logistic', early_stopping=True, epsilon=1e-05)
mlp1.fit(X_train,y_train)

predictions = mlp1.predict(X_test)
print(classification_report(y_test,predictions))
#print(confusion_matrix(y_test,predictions))


# 25% of data for testing, using smooth/soft logistic sigmoid function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=42)
scaler2 = StandardScaler()
scaler2.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler2.transform(X_train)
X_test = scaler2.transform(X_test)

mlp2 = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000, activation='logistic', early_stopping=True, epsilon=1e-05)
mlp2.fit(X_train,y_train)
predictions2 = mlp2.predict(X_test)
print(classification_report(y_test,predictions2))
#print(confusion_matrix(y_test,predictions2))


# 75% data testing, using hard relu activation function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, train_size=0.25, random_state=42)


scaler3 = StandardScaler()
scaler3.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler3.transform(X_train)
X_test = scaler3.transform(X_test)

mlp3 = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000, activation='relu', early_stopping=True, epsilon=1e-05)
mlp3.fit(X_train,y_train)

predictions3 = mlp3.predict(X_test)
print(classification_report(y_test,predictions3))
#print(confusion_matrix(y_test,predictions3))

# 25% data testing, using hard relu activation function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=42)


scaler4 = StandardScaler()
scaler4.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler4.transform(X_train)
X_test = scaler4.transform(X_test)

mlp4 = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000, activation='relu', early_stopping=True, epsilon=1e-05)
mlp4.fit(X_train,y_train)

predictions4 = mlp4.predict(X_test)
print(classification_report(y_test,predictions4))
#print(confusion_matrix(y_test,predictions4))
