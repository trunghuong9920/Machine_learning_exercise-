import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Pima diabetes dataset from CSV
# and convert into a NumPy matrix suitable for
# extraction into X, y format needed for Scikit-Learn
diabetes = pd.read_csv('water.csv').values

# Extract the feature columns and outcome response
# into appropriate variables
X = diabetes[:, 1:10]
y = diabetes[:, 10]

# Create and fit a perceptron model (with reproducible
# random seed)
trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3)
model = Perceptron(random_state=1)
model.fit(trainX, trainY)

# Output the (in sample) mean accuracy score
# of the classification
# print("%0.3f" % model.score(X, y))

print("Xtest = ", testX)
print("ytest = ", testY)
y_pred = model.predict(testX)
print("predict= ",y_pred)
print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
    
