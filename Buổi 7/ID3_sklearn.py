import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

from id3 import entropy

# Outlook={'Sunny':0,'Overcast':1,'Rain':2}
# Temp={'Hot':0,'Mild':1,'Cool':2}
# Humidity={'High':0,'Normal':1}
# Wind={'Weak':0,'Strong':1}
# Decision={'Yes':0,'No':1}

# csv['Outlook']=csv['Outlook'].map(Outlook)
# csv['Temp']=csv['Temp'].map(Temp)
# csv['Humidity']=csv['Humidity'].map(Humidity)
# csv['Wind']=csv['Wind'].map(Wind)
# csv['Decision']=csv['Decision'].map(Decision)

# cotdt=['Outlook','Temp','Humidity','Wind']
# x=csv[cotdt]
# y=csv['Decision']

diabetes = pd.read_csv('B2_test.csv').values

X = diabetes[:, 1:4]
y = diabetes[:, 4]

# trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3)
# Tính theo thuật toán entropy
clf = tree.DecisionTreeClassifier(criterion = 'entropy')           
clf = clf.fit(X, y)

print("%0.3f" % clf.score(X, y))

y_pred = clf.predict(X)
# print("y_test= ", testY)
print("predict= ",y_pred)

print('Accuracy: {:.2f}'.format(accuracy_score(y, y_pred)))

# name = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
# print(clf)
# fig=plt.figure(figsize=(25,20))
# _=tr.plot_tree(clf,class_names=name)
# fig.savefig('cay.png')


