import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from tkinter import *
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

    
#------------load and read file csv--------------------
diabetes = pd.read_csv('data.csv').values

X = diabetes[:, :15]
y = diabetes[:, 15]

y = np.where(y == "YES", "2", y)
y = np.where(y == "NO", "1", y)

X = np.where(X == "M", "1", X)
X = np.where(X == "F", "2", X)

# Standardize the Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA Projection to 2D
pca = PCA(n_components=2)
Xpca = pca.fit_transform(X)


# split data to training and testing 
trainX, testX, trainY, testY = train_test_split(Xpca, y, test_size = 0.3, random_state=42)

# # ------------predict CART model--------------------
model = tree.DecisionTreeClassifier(criterion = 'gini')
model.fit(trainX, trainY)

# ---------checking ytest and y_predic---------
y_pred = model.predict(testX)

for i in range(len(y_pred)):
    print("Kết quả thực tế: ",testY[i]," =>Kết quả dự đoán: ",y_pred[i])

# ---------Precision--------------
precision = round(precision_score(testY, y_pred, average='micro') * 100,2)
recall = round(recall_score(testY, y_pred, average='micro') * 100,2)
f1_scor = round(f1_score(testY, y_pred, average='micro') * 100,2)

print("\nĐộ chính xác precision : ", precision)
print("\nĐộ chính xác recall: ", recall)
print("\nĐộ chính xác f1_score: ", f1_scor)



# # -------predic----------
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
    global t10
    global t11
    global t12
    global t13
    global t14
    global t15

    t1 = float(var1.get())
    t2 = float(entry2.get())
    t3 = float(var2.get())
    t4 = float(var3.get())
    t5 = float(var4.get())
    t6 = float(var5.get())
    t7 = float(var6.get())
    t8 = float(var7.get())
    t9 = float(var8.get())
    t10 = float(var9.get())
    t11 = float(var10.get())
    t12 = float(var11.get())
    t13 = float(var12.get())
    t14 = float(var13.get())
    t15 = float(var14.get())
    
    Xip = np.array([[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15]])

    Xip= scaler.transform(Xip)
    Xip=pca.transform(Xip)
    
    resul = model.predict(Xip)
    if resul == "2":
        resultEnd = "Yes"
    else:
        resultEnd = "No"

    
    label_show.set(resultEnd)


    
    
root =Tk()
root.option_add("*Font","TimeNewRoman 14")
label_show=StringVar()
label_show1=StringVar()
label_show2=StringVar()
label_show3=StringVar()
var1 = StringVar(value=1)
var2 = StringVar(value=1)
var3 = StringVar(value=1)
var4 = StringVar(value=1)
var5 = StringVar(value=1)
var6 = StringVar(value=1)
var7 = StringVar(value=1)
var8 = StringVar(value=1)
var9 = StringVar(value=1)
var10 = StringVar(value=1)
var11= StringVar(value=1)
var12 = StringVar(value=1)
var13 = StringVar(value=1)
var14 = StringVar(value=1)

Label (root, text="Hãy nhập thông tin(nếu có biểu hiện gì ở dưới vui lòng tích vào ô)").grid(row=0,columnspan=2)

Label (root, text="GENDER").grid(row=1,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root, text='Tích vào nếu là Nữ',variable=var1, onvalue=2, offvalue=1).grid(row=1,column=1,padx=10,pady=10,sticky = W)

Label (root, text="AGE").grid(row=2,column=0,padx=10,pady=10,sticky = W)
entry2 = Entry(root,text="20")
entry2.grid(row=2,column=1,padx=10)

Label (root, text="SMOKING").grid(row=3,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var2, onvalue=2, offvalue=1).grid(row=3,column=1,padx=10,pady=10,sticky = W)

Label (root, text="YELLOW_FINGERS").grid(row=4,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var3, onvalue=2, offvalue=1).grid(row=4,column=1,padx=10,pady=10,sticky = W)

Label (root, text="ANXIETY").grid(row=5,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var4, onvalue=2, offvalue=1).grid(row=5,column=1,padx=10,pady=10,sticky = W)

Label (root, text="PEER_PRESSURE").grid(row=6,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var5, onvalue=2, offvalue=1).grid(row=6,column=1,padx=10,pady=10,sticky = W)

Label (root, text="CHRONIC DISEASE").grid(row=7,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var6, onvalue=2, offvalue=1).grid(row=7,column=1,padx=10,pady=10,sticky = W)

Label (root, text="FATIGUE").grid(row=8,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var7, onvalue=2, offvalue=1).grid(row=8,column=1,padx=10,pady=10,sticky = W)

Label (root, text="ALLERGY").grid(row=1,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var8, onvalue=2, offvalue=1).grid(row=1,column=3,padx=10,pady=10,sticky = W)

Label (root, text="WHEEZING").grid(row=2,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var9, onvalue=2, offvalue=1).grid(row=2,column=3,padx=10,pady=10,sticky = W)

Label (root, text="ALCOHOL CONSUMING").grid(row=3,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var10, onvalue=2, offvalue=1).grid(row=3,column=3,padx=10,pady=10,sticky = W)

Label (root, text="COUGHING").grid(row=4,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var11, onvalue=2, offvalue=1).grid(row=4,column=3,padx=10,pady=10,sticky = W)

Label (root, text="SHORTNESS OF BREATH").grid(row=5,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var12, onvalue=2, offvalue=1).grid(row=5,column=3,padx=10,pady=10,sticky = W)

Label (root, text="SWALLOWING DIFFICULTY").grid(row=6,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var13, onvalue=2, offvalue=1).grid(row=6,column=3,padx=10,pady=10,sticky = W)

Label (root, text="CHEST PAIN").grid(row=7,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var14, onvalue=2, offvalue=1).grid(row=7,column=3,padx=10,pady=10,sticky = W)

Button (root, text="Kết quả dự đoán", command=doan,bg= 'cyan').grid(row=8,columnspan=2,padx=10,pady=10,sticky = E)

Label (root, text="Cart:").grid(row=9,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show).grid(row=9,column=2,padx=10,pady=10,sticky = E)

Label (root, text="Tỉ lệ dự đoán presicion:").grid(row=10,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show1).grid(row=10,column=2,padx=10,pady=10,sticky = E)

Label (root, text="Tỉ lệ dự đoán recall:").grid(row=11,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show2).grid(row=11,column=2,padx=10,pady=10,sticky = E)

Label (root, text="Tỉ lệ dự đoán f1-score:").grid(row=12,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show3).grid(row=12,column=2,padx=10,pady=10,sticky = E)

label_show1.set(precision)
label_show2.set(recall)
label_show3.set(f1_scor)

root.mainloop()
