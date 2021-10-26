# solution by sklearn
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np


X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

#diabetes = pd.read_csv('water.csv').values
#X = diabetes[:, 1:10]
#y = diabetes[:, 10]


model = SVC(kernel = 'linear', C = 1e5) # just a big number
model.fit(X, y)
w = model.coef_
b = model.intercept_
y_pred = model.predict(X)
rate = accuracy_score(y, y_pred)*100

print('w = ', w)
print('b = ', b)

print("predict= ",y_pred)
print("accuracy_score= ",rate,"%")

