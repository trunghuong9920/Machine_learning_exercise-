import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = [[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]
K = 3

model = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(model.cluster_centers_)
pred_label = model.predict(X)
print(pred_label)
#kmeans_display(X, pred_label, 'res_scikit.pdf')
