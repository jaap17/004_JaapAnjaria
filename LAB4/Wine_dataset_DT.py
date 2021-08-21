from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

#loading dataset
dataset = datasets.load_wine()
X = dataset.data
print(X)
y = dataset.target


#splitting test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Intializing the decsion tree Classifier
model = tree.DecisionTreeClassifier()
#training the data
model.fit(X_train, y_train)
print(model)

#predict the output
expected_y  = y_test
predicted_y = model.predict(X_test)

#Printing classification of the model and its confusion matrix
print(metrics.classification_report(expected_y, predicted_y, target_names=dataset.target_names))
print(metrics.confusion_matrix(expected_y, predicted_y))


from sklearn.tree import export_graphviz
export_graphviz(model,out_file='wine_tree.dot',feature_names=list(dataset.feature_names),
               class_names=list(dataset.target_names), filled=True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'wine_tree.dot', '-o', 'wine_tree.png', '-Gdpi=600'])

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('wine_tree.png'))
plt.axis('off')
plt.show()