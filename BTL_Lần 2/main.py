from numpy.lib.function_base import percentile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.tree as tree
from sklearn.linear_model import Perceptron


#------------load and read file csv--------------------
diabetes = pd.read_csv('water.csv').values

X = diabetes[:, 1:10]
y = diabetes[:, 10]
# print("X",X)
# print("y",y)

# -----------------Diabe data training Trainning and testing
trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3, shuffle=False)

#-----------------ID3---------------------------------
clf_id3 = tree.DecisionTreeClassifier(criterion = 'entropy')
clf_id3.fit(trainX, trainY)

#------------------perceptron-----------------
clf_pec = Perceptron()
clf_pec.fit(trainX, trainY)

y_pred_id3 = clf_id3.predict(testX)
y_pred_pec = clf_pec.predict(testX)

for i in range(60):
    print("Giá trị thực: ",testY[i],"=> Dự đoán ID3: ",y_pred_id3[i],"=>Dự đoán Pec: ", y_pred_pec[i])

rate_id3 = accuracy_score(testY, y_pred_id3)*100
rate_pec = accuracy_score(testY, y_pred_pec)*100
print('Accuracy_id3: ',rate_id3)
print('Accuracy_pec: ', rate_pec)


# # name = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']

# # print(clf)
# # fig=plt.figure(figsize=(25,20))
# # _=tr.plot_tree(clf,class_names=name)
# # fig.savefig('cay.png')

