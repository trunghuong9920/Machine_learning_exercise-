import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.metrics import accuracy_score
from tkinter import *


def PCA_projection(X,scaler):
    # Standardize the Data

    X = scaler.fit_transform(X)

    # PCA Projection to 2D
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform (X)
    return Xpca
def predict(X,y,scaler):
    # -------PCA_projection------------
    Xpca= PCA_projection(X,scaler)
    trainX = Xpca[:218, :]
    trainY = y[:218]
    testX = Xpca[218 : , :]
    testY = y[218:]
    # trainX, testX, trainY, testY = train_test_split(Xpca, y, test_size = 0.3, random_state=1)
    model = tree.DecisionTreeClassifier(criterion = 'gini')
    model.fit(trainX, trainY)
    y_pred = model.predict(testX)
    return y_pred,testY,X

    
#------------load and read file csv--------------------
diabetes = pd.read_csv('ungthu.csv').values

X = diabetes[:, :15]
y = diabetes[:, 15]


# ------------predict--------------------
scaler=StandardScaler()

y_pred,testY,X = predict(X,y,scaler)
# ---------checking ytest and y_predic---------
for i in range(len(testY)):
    print("Kết quả thực tế: ",testY[i]," =>Kết quả dự đoán: ",y_pred[i])

# ---------Precision--------------
rate = (accuracy_score(testY,y_pred) * 100)
print("\nĐộ chính xác: ", rate)


# y_pred,testY,X = predict(Xnew,y)

#-------------------------------View predict--------------------


