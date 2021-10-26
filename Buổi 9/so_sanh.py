# solution by sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sklearn.tree as tree
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np


# X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
# 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
# y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

diabetes = pd.read_csv('B1_test.csv').values
X = diabetes[:, 1:5]
y = diabetes[:, 5]


svm = SVC(kernel = 'linear', C = 1e5) # just a big number
logre = LogisticRegression()
clf = tree.DecisionTreeClassifier(criterion = 'entropy')

svm.fit(X, y)
logre.fit(X, y)
clf.fit(X, y)

# w = svm.coef_
# b = svm.intercept_
y_pred_svm = svm.predict(X)
y_pred_log = logre.predict(X)
y_pred_clf = clf.predict(X)

rate_svm = accuracy_score(y, y_pred_svm)*100
rate_clf = accuracy_score(y, y_pred_clf)*100
rate_log = accuracy_score(y, y_pred_log)*100


# print('w = ', w)
# print('b = ', b)

print("predict= ",y_pred_svm)
print("predict= ",y_pred_log)
print("predict= ",y_pred_clf)

print("accuracy_score_svm= ",rate_svm,"%")
print("accuracy_score_id3= ",rate_clf,"%")
print("accuracy_score_log= ",rate_log,"%")



