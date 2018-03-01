# This program will verify if a fruit is an apple or an orange
# by looking at his texture and weigth

from sklearn import tree

# Feature is a combination of weight and texture.
# The texture is represented by 0(bumpy) or 1(smooth)
features = [[140, 1],[130,1],[150,0],[170,0]]

#In labels, 0 means apple and 1 means orange
labels = [0,0,1,1]

#clf means classfier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# It's heavy and bumpy, so probaly is an Orange
# We expected that line 19 prints an orange
print (clf.predict([[150,0]]))