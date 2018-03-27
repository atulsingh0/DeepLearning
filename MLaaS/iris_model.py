
# coding: utf-8

# ## Machine Learning Model as a Service

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

import numpy as np
import pandas as pd

# Loading data
iris_data = load_iris()
X, y = iris_data['data'], iris_data['target']
columns = iris_data['feature_names']


# Suffling and Splitting the dataset into train and test
dataset = np.hstack((X, y.reshape(-1, 1)))
np.random.shuffle(dataset)
X_train, X_test, y_train, y_test = train_test_split(dataset[:, :4], dataset[:, 4], test_size=0.2)

# Model
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# printing Model accuracy
print("Accuracy Score : ", accuracy_score(y_test, y_pred))

# Saving the model as file
joblib.dump(clf, 'iris.model')
