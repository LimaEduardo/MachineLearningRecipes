# Here we're gonna use the Iris data set 
# https://en.wikipedia.org/wiki/Iris_flower_data_set

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

#print (iris.feature_names)
#print (iris.target_names)
#print (iris.data[0])
#print (iris.target[0])

#for i in range(len(iris.target)):
#	print ("Example %d: label %s, features %s " % (i,iris.target[i], iris.data[i]))

# Remove an example of each flower to test later
test_index = [0,50,100]

#Training data
train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis=0)

#Testing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))

# Vizualization code that prints decision tree
from sklearn.externals.six import StringIO
import pydotplus


dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,
                         special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")