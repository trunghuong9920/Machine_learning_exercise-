from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import pandas as pd

data = pd.read_csv(r"wt_train.csv")
x1 = np.array(data[['ph']])
x2= np.array(data[['Hardness']])
x3 = np.array(data[['Solids']])
x4 = np.array(data[['Chloramines']])
y = np.array(data[['Potability']]).T[0]


X = np.concatenate((x1,x2,x3,x4),axis=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one,X),axis=1)
print(X)
print(Xbar)
def predict(w, X):
    return np.sign(X.dot(w))

def perceptron(X, y, w_init):
    w = w_init
    while True:
        pred = predict(w, X)
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        num_mis = mis_idxs.shape[0]
        if num_mis == 0: 
            return w
        random_id = np.random.choice(mis_idxs, 1)[0]
        w = w + y[random_id]*X[random_id]
    return w


w_init1 = 0.1*np.random.randn(Xbar.shape[1])
w1 = perceptron(Xbar, y, w_init1)
print(w1)
