#Import scikit-learn dataset library
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

#Load dataset
iris = datasets.load_iris()
X = iris['data']
# print the names of the 13 features
print(list(iris.keys()))

# print the label type of wine(class_0, class_1, class_2)
print(iris['target_names'])

# print data(feature)shape
print(iris['data'].shape)

#import the necessary module
from sklearn.model_selection import train_test_split

#split data set into train and test sets
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=50, test_size=0.25)

#Create a Decision Tree Classifier (using Gini)
clf = DecisionTreeClassifier(criterion='gini')


#Train the model using the training sets
clf.fit(X_train,y_train)

# Predict the classes of test data
y_pred = clf.predict(X_test)

print(y_pred)


from sklearn.tree import export_graphviz
export_graphviz(clf,out_file='iris_tree.dot',feature_names=list(iris.feature_names),
               class_names=list(iris.target_names), filled=True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'iris_tree.dot', '-o', 'iris_tree.png', '-Gdpi=600'])

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('iris_tree.png'))
plt.axis('off')
plt.show()