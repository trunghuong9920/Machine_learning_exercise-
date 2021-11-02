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



def PCA_projection(X,scaler,pca):
    # Standardize the Data
    X = scaler.fit_transform(X)

    # PCA Projection to 2D
    
    Xpca = pca.fit_transform (X)
    return Xpca
def predict(X,y,scaler,pca):
    # -------PCA_projection------------
    Xpca= PCA_projection(X,scaler,pca)
    
    trainX, testX, trainY, testY = train_test_split(Xpca, y, test_size = 0.3, random_state=1)
    model = tree.DecisionTreeClassifier(criterion = 'gini')
    model.fit(trainX, trainY)
    y_pred = model.predict(testX)
    return y_pred,testY,model

    
#------------load and read file csv--------------------
diabetes = pd.read_csv('ungthu.csv').values

X = diabetes[:, :15]
y = diabetes[:, 15]


# ------------predict--------------------
scaler = StandardScaler()
pca = PCA(n_components=2)
y_pred,testY,model = predict(X,y,scaler,pca)

# ---------checking ytest and y_predic---------
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

    if var1.get() == '':
        t1= 1
    else:
        t1 = float(var1.get()) 
    if entry2.get() == '':
        t2= 10
    else:
        t2 = float(entry2.get())
    if var2.get() == '':
        t3= 1
    else:
        t3 = float(var2.get())
    if var4.get() == '':
        t4= 1
    else:
        t4 = float(var3.get())
    if var4.get() == '':
        t5= 1
    else:
        t5 = float(var4.get())
    if var5.get() == '':
        t6= 1
    else:
        t6 = float(var5.get())
    if var6.get() == '':
        t7= 1
    else:
        t7 = float(var6.get())
    if var7.get() == '':
        t8= 2
    else:
        t8 = float(var7.get())
    if var8.get() == '':
        t9= 1
    else:
        t9 = float(var8.get())
    if var9.get() == '':
        t10= 1
    else:
        t10 = float(var9.get())
    if var10.get() == '':
        t11= 1
    else:
        t11 = float(var10.get())
    if var11.get() == '':
        t12= 1
    else:
        t12 = float(var11.get())
    if var12.get() == '':
        t13= 1
    else:
        t13 = float(var12.get())
    if var13.get() == '':
        t14= 1
    else:
        t14 = float(var13.get())
    if var14.get() == '':
        t15= 1
    else:
        t15 = float(var14.get())
    
    Xip = np.array([[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15]])

    Xip= scaler.transform(Xip)
    Xip=pca.transform(Xip)
    
    resul = model.predict(Xip)
    if resul == 2:
        resultEnd = "Yes"
    else:
        resultEnd = "No"

    
    label_show.set(resultEnd)
    label_show1.set(precision)
    label_show2.set(recall)
    label_show3.set(f1_scor)

    
    
root =Tk()
root.option_add("*Font","TimeNewRoman 14")
label_show=StringVar()
label_show1=StringVar()
label_show2=StringVar()
label_show3=StringVar()
var1 = StringVar()
var2 = StringVar()
var3 = StringVar()
var4 = StringVar()
var5 = StringVar()
var6 = StringVar()
var7 = StringVar()
var8 = StringVar()
var9 = StringVar()
var10 = StringVar()
var11= StringVar()
var12 = StringVar()
var13 = StringVar()
var14 = StringVar()

Label (root, text="Hãy nhập thông tin").grid(row=0,columnspan=2)
Label (root, text="GENDER").grid(row=1,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root, text='Male/Female',variable=var1, onvalue=1, offvalue=2).grid(row=1,column=1,padx=10,pady=10,sticky = W)
Label (root, text="AGE").grid(row=2,column=0,padx=10,pady=10,sticky = W)
entry2 = Entry(root,text="20")
entry2.grid(row=2,column=1,padx=10)
Label (root, text="SMOKING").grid(row=3,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var2, onvalue=1, offvalue=2).grid(row=3,column=1,padx=10,pady=10,sticky = W)
Label (root, text="YELLOW_FINGERS").grid(row=4,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var3, onvalue=1, offvalue=2).grid(row=4,column=1,padx=10,pady=10,sticky = W)
Label (root, text="ANXIETY").grid(row=5,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var4, onvalue=1, offvalue=2).grid(row=5,column=1,padx=10,pady=10,sticky = W)
Label (root, text="PEER_PRESSURE").grid(row=6,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var5, onvalue=1, offvalue=2).grid(row=6,column=1,padx=10,pady=10,sticky = W)
Label (root, text="CHRONIC DISEASE").grid(row=7,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var6, onvalue=1, offvalue=2).grid(row=7,column=1,padx=10,pady=10,sticky = W)
Label (root, text="FATIGUE").grid(row=8,column=0,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var7, onvalue=1, offvalue=2).grid(row=8,column=1,padx=10,pady=10,sticky = W)
Label (root, text="ALLERGY").grid(row=1,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var8, onvalue=1, offvalue=2).grid(row=1,column=3,padx=10,pady=10,sticky = W)
Label (root, text="WHEEZING").grid(row=2,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var9, onvalue=1, offvalue=2).grid(row=2,column=3,padx=10,pady=10,sticky = W)
Label (root, text="ALCOHOL CONSUMING").grid(row=3,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var10, onvalue=1, offvalue=2).grid(row=3,column=3,padx=10,pady=10,sticky = W)
Label (root, text="COUGHING").grid(row=4,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var11, onvalue=1, offvalue=2).grid(row=4,column=3,padx=10,pady=10,sticky = W)
Label (root, text="SHORTNESS OF BREATH").grid(row=5,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var12, onvalue=1, offvalue=2).grid(row=5,column=3,padx=10,pady=10,sticky = W)
Label (root, text="SWALLOWING DIFFICULTY").grid(row=6,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var13, onvalue=1, offvalue=2).grid(row=6,column=3,padx=10,pady=10,sticky = W)
Label (root, text="CHEST PAIN").grid(row=7,column=2,padx=10,pady=10,sticky = W)
Checkbutton(root,variable=var14, onvalue=1, offvalue=2).grid(row=7,column=3,padx=10,pady=10,sticky = W)
Button (root, text="Kết quả dự đoán", command=doan,bg= 'cyan').grid(row=8,columnspan=2,padx=10,pady=10,sticky = E)
Label (root, text="Cart:").grid(row=9,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show).grid(row=9,column=2,padx=10,pady=10,sticky = E)
Label (root, text="Tỉ lệ dự đoán presicion:").grid(row=10,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show1).grid(row=10,column=2,padx=10,pady=10,sticky = E)
Label (root, text="Tỉ lệ dự đoán recall:").grid(row=11,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show2).grid(row=11,column=2,padx=10,pady=10,sticky = E)
Label (root, text="Tỉ lệ dự đoán f1-score:").grid(row=12,columnspan=2,padx=10,pady=10,sticky = W)
Label(root,textvariable=label_show3).grid(row=12,column=2,padx=10,pady=10,sticky = E)
Label (root, text="Chú thích: check = 1 / not check = 2 ").grid(row=13,columnspan=2,padx=10,pady=10,sticky = W)

root.mainloop()
