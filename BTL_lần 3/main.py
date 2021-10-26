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

# ---------------------------TRANING AND TEST-------------------------------
trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3,random_state=1)

svm = SVC(kernel = 'linear', C = 1e5) # just a big number
logre = LogisticRegression()

svm.fit(trainX,trainY)
wSvm_0=svm.intercept_
wSvm_1=svm.coef_

logre.fit(trainX, trainY)
wLog_0=logre.intercept_
wLog_1=logre.coef_

#--------------------------RESULT--------------------------------------------------
y_pred_svm = svm.predict(testX)
y_pred_log = logre.predict(testX)
rate_svm = accuracy_score(testY, y_pred_svm)*100
rate_log = accuracy_score(testY, y_pred_log)*100

for i in range(60):
    print("Gía trị thưc: ",testY[i]," => Dự đoán svm: ",y_pred_svm[i]," =>Dự đoán log: ",y_pred_log[i])
print("\nRate_SVM= ",rate_svm)
print("Rate_log= ",rate_log)

#-------------------------------View predict--------------------
def doan():
    global t1
    global t2
    global t3
    global t4
    global t5
    global t6
    global t7
    global t8
    global t9

    t1 = float(entry1.get())
    t2 = float(entry2.get())
    t3 = float(entry3.get())
    t4 = float(entry4.get())
    t5 = float(entry5.get())
    t6 = float(entry6.get())
    t7 = float(entry7.get())
    t8 = float(entry8.get())
    t9 = float(entry9.get())

    a = np.array([t1,t2,t3,t4,t5,t6,t7,t8,t9]).T
    ySvm = np.dot(wSvm_1 , a) + wSvm_0
    yLog = np.dot(wLog_1 , a) + wLog_0

    if ySvm < 0:
        label_show.set(float(0))
    else:
        label_show.set(float(1))
    if yLog < 0:
        label_show1.set(float(0))
    else:
        label_show1.set(float(1))
    
    
root =Tk()
root.option_add("*Font","TimeNewRoman 14")
label_show=StringVar()
label_show1=StringVar()

Label (root, text="Hãy nhập thông tin").grid(row=0,columnspan=2)
Label (root, text="ph").grid(row=1,column=0,padx=10,pady=10,sticky = W)
entry1= Entry(root) 
entry1.grid(row=1,column=1,padx=10)
Label (root, text="Hardness").grid(row=2,column=0,padx=10,pady=10,sticky = W)
entry2 = Entry(root)
entry2.grid(row=2,column=1,padx=10)
Label (root, text="Solids").grid(row=3,column=0,padx=10,pady=10,sticky = W)
entry3 = Entry(root)
entry3.grid(row=3,column=1,padx=10)
Label (root, text="Chloramines").grid(row=4,column=0,padx=10,pady=10,sticky = W)
entry4 = Entry(root)
entry4.grid(row=4,column=1,padx=10)
Label (root, text="Sulfate").grid(row=5,column=0,padx=10,pady=10,sticky = W)
entry5 = Entry(root)
entry5.grid(row=5,column=1,padx=10)
Label (root, text="Conductivity").grid(row=6,column=0,padx=10,pady=10,sticky = W)
entry6 = Entry(root)
entry6.grid(row=6,column=1,padx=10)
Label (root, text="Organic_carbon").grid(row=7,column=0,padx=10,pady=10,sticky = W)
entry7 = Entry(root)
entry7.grid(row=7,column=1,padx=10)
Label (root, text="Trihalomethanes").grid(row=8,column=0,padx=10,pady=10,sticky = W)
entry8 = Entry(root)
entry8.grid(row=8,column=1,padx=10)
Label (root, text="urbidity").grid(row=9,column=0,padx=10,pady=10,sticky = W)
entry9 = Entry(root)
entry9.grid(row=9,column=1,padx=10)
Button (root, text="Kết quả dự đoán", command=doan,bg= 'cyan').grid(row=10,column=0,padx=10,pady=10,sticky = E)
Label (root, text="SVM").grid(row=11,column=0,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show).grid(row=11,column=1)
Label (root, text="Logistic Regestion").grid(row=12,column=0,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show1).grid(row=12,column=1)
root.mainloop()
