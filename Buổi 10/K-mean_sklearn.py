import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


diabetes = pd.read_csv('ungthu.csv').values

X = diabetes[:, 1:15]
# X = [[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]
y = diabetes[:, 15]

X = StandardScaler().fit_transform(X)
print(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)

# print(X)

# print(pca.singular_values_)
# K = 1

model = KMeans(n_clusters=2, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(model.cluster_centers_)
pred_label = model.predict(X)

for i in range(200):
    print("y thực tế: ", y[i],", y dự đoán: ",pred_label[i])

print(accuracy_score(y,pred_label))
#kmeans_display(X, pred_label, 'res_scikit.pdf')
