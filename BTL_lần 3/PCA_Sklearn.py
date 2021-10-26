import pandas as pd
import numpy as np
from tkinter import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#------------------getDATA---------------------
diabetes = pd.read_csv('data_water.csv').values

X = diabetes[:, 1:10]
y = diabetes[:, 10]

from sklearn.preprocessing import StandardScaler

# Standardize the features
X = StandardScaler().fit_transform(X)
# Import PCA from sklearn
from sklearn.decomposition import PCA

# Instantiate PCA
pca = PCA(n_components=2)

# Fit PCA to features
principalComponents = pca.fit_transform(X)
# Create a new dataset from principal components 
df = pd.DataFrame(data = principalComponents, 
                  columns = ['PC1', 'PC2'])

# target = pd.Series(iris['target'], name='target')

# result_df = pd.concat([df, target], axis=1)
# result_df.head(5)
# print(df)
log = LogisticRegression()
log.fit(df,y)
ypr = log.predict(df)
print(ypr)