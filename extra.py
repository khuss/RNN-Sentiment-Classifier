# Train from scratch
from sklearn import svm
from sklearn import datasets
clf = svm.SVC(C=1, kernel='linear')
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)


# import pickle
from joblib import dump, load
filename = 'iris.p'

dump(clf, filename)
clf2 = load(filename)
clf2.predict(X[0:1])
