from __future__ import division, print_function, unicode_literals
from sklearn import datasets, linear_model
import numpy as np
import pandas as pd
from tkinter import *
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


dt = pd.read_csv("datapricemoto.csv")
print("Dataset.head()\n",dt.head())
X1 = np.array(dt['ex_showroom_price'])
X2 = np.array(dt['year'])
X3 = np.array(dt['km_driven'])
X44 = np.array(dt['owner'])
X4 = []
for i in X44:
    X4.append(np.array(int(i[0][0])))

x = np.array([X1,X2,X4,X3]).T
y = np.array(dt[['selling_price']])

# Dividing input and output data to train and test data
# Training : Testing = 70 : 30

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, 
                                                     random_state = 0)

regressor = linear_model.LinearRegression()
regressor.fit(xtrain, ytrain)



print(xtrain)

y_pred = regressor.predict(xtest)
y_pred1 = y_pred
y_pred1 = y_pred1.reshape(-1,1)

print("\n   RESULT OF LINEAR REGRESSION PREDICTION : ")
print ("\nPredicted value : \n", y_pred1)
print('scikit-learn’s solution : w_1 = ', regressor.coef_, 'w_0 = ', regressor.intercept_)
w_0=regressor.intercept_
w_1=regressor.coef_

for i in range(45):
    print ("Predicted value : ", y_pred1[i],"real value: ", ytest[i], "% sai khac: ",abs(y_pred1[i]-ytest[i])/ytest[i]*100 )

# cv= 0
# for i in range(148):
#     delta = (y[i] - (w_1*x[i] + w_0))
#     cv = cv+ delta*delta
# print("cross validation = ", cv)


def doan():
    global t1
    global t2
    global t3
    global t4

    t1 = float(entry1.get())
    t2 = float(entry2.get())
    t3 = float(entry3.get())
    t4 = float(entry4.get())

    a = np.array([t1,t2,t3,t4]).T
    y8 = np.dot(w_1 , a) + w_0

    label_show.set(float(y8))

root =Tk()
root.option_add("*Font","TimeNewRoman 14")
label_show=StringVar()
Label (root, text="Hãy nhập thông tin").grid(row=0,columnspan=2)
Label (root, text="Giá bán tại showroom").grid(row=1,column=0,padx=10,pady=10,sticky = W)
entry1= Entry(root) 
entry1.grid(row=1,column=1,padx=10)
Label (root, text="Năm bán").grid(row=2,column=0,padx=10,pady=10,sticky = W)
entry2 = Entry(root)
entry2.grid(row=2,column=1,padx=10)
Label (root, text="Số chủ sở hữu trước đó").grid(row=3,column=0,padx=10,pady=10,sticky = W)
entry3 = Entry(root)
entry3.grid(row=3,column=1,padx=10)
Label (root, text="Số km đã chạy").grid(row=4,column=0,padx=10,pady=10,sticky = W)
entry4 = Entry(root)
entry4.grid(row=4,column=1,padx=10)
Button (root, text="Kết quả dự đoán", command=doan,bg= 'cyan').grid(row=7,column=0,padx=10,pady=10,sticky = E)
Label(root,textvariable=label_show).grid(row=7,column=1)
root.mainloop()



plt.plot(y_pred)
plt.plot(ytest)
plt.xlabel("Xe")
plt.ylabel('Giá tiền')
plt.title('Biểu đồ so sánh giá trị dự đoán và giá trị thực tế')
plt.legend(['Giá trị dự đoán', 'Giá trị thực tế'])
plt.show()
