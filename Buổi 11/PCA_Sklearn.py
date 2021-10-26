import numpy as np

np.random.seed(1)
states = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight','sad', 'sleepy', 'surprised', 'wink' ]
h, w, K = 116, 98, 100 # hight, weight, new dim
D = h * w
N = len(states)*15
X = np.zeros((D, N))

# Doing PCA, note that each row is a datapoint
from sklearn.decomposition import PCA
pca = PCA(n_components=K) # K = 100
pca.fit(X.T)
# projection matrix
U = pca.components_.T
print(U)