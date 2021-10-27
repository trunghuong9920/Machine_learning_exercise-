from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("indian_liver_patient_dataset_fix.csv")
X = df.drop(['class'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

form = Tk()
form.title("Dự đoán người bị mắc bệnh gan:")
form.geometry("600x600")

lb_ten = Label(form, text="Nhập thông tin bệnh nhân:")
lb_ten.grid(row=1, column=2, padx=40, pady=10)

lb_age = Label(form, text="Tuổi của bệnh nhân tính theo năm (age)")
lb_age.grid(row=2, column=2, padx=40, pady=10)
tb_age = Entry(form)
tb_age.grid(row=2, column=3)

lb_gender = Label(form, text="Giới tính bệnh nhân: Nam hoặc Nữ (gender)")
lb_gender.grid(row=3, column=2, pady=10)
tb_gender = Entry(form)
tb_gender.grid(row=3, column=3)

lb_TB = Label(form, text="Bilirubin toàn phần (TB)")
lb_TB.grid(row=4, column=2, pady=10)
tb_TB = Entry(form)
tb_TB.grid(row=4, column=3)

lb_DB = Label(form, text="Bilirubin trực tiếp (DB)")
lb_DB.grid(row=5, column=2, pady=10)
tb_DB = Entry(form)
tb_DB.grid(row=5, column=3)

lb_alkphos = Label(form, text="Phosphosite kiềm (alkphos)")
lb_alkphos.grid(row=6, column=2, pady=10)
tb_alkphos = Entry(form)
tb_alkphos.grid(row=6, column=3)

lb_sgpt = Label(form, text="Alanine Aminotransferase (sgpt)")
lb_sgpt.grid(row=7, column=2, pady=10)
tb_sgpt = Entry(form)
tb_sgpt.grid(row=7, column=3)

lb_sgot = Label(form, text="Aspartate Aminotransferase (sgot)")
lb_sgot.grid(row=8, column=2, pady=10)
tb_sgot = Entry(form)
tb_sgot.grid(row=8, column=3)

lb_TP = Label(form, text="Tổng số protein (TP) ")
lb_TP.grid(row=9, column=2, pady=10)
tb_TP = Entry(form)
tb_TP.grid(row=9, column=3)

lb_ALB = Label(form, text="Albumin (ALB)")
lb_ALB.grid(row=10, column=2, pady=10)
tb_ALB = Entry(form)
tb_ALB.grid(row=10, column=3)

lb_A_G = Label(form, text="Tỷ lệ Albumin và Globulin (A_G):")
lb_A_G.grid(row=11, column=2, pady=10)
tb_A_G = Entry(form)
tb_A_G.grid(row=11, column=3)


def logistic():
    age = tb_age.get()
    gender = tb_gender.get()
    TB = tb_TB.get()
    DB = tb_DB.get()
    alkphos = tb_alkphos.get()
    sgqt = tb_sgpt.get()
    sgot = tb_sgot.get()
    TP = tb_TP.get()
    ALB = tb_ALB.get()
    A_G = tb_A_G.get()
    X_pred = np.array([age, gender, TB, DB, alkphos, sgqt, sgot, TP, ALB, A_G], dtype=float).reshape(1, -1)
    y_logitic = logisticRegression.predict(X_pred)
    messagebox.showinfo("Kết quả dự đoán theo logiticRegression",
                        "Người bị bệnh gan (1 là bị nhiễm) , (2 là không bị ): " + str(y_logitic))


def svm():
    age = tb_age.get()
    gender = tb_gender.get()
    TB = tb_TB.get()
    DB = tb_DB.get()
    alkphos = tb_alkphos.get()
    sgpt = tb_sgpt.get()
    sgot = tb_sgot.get()
    TP = tb_TP.get()
    ALB = tb_ALB.get()
    A_G = tb_A_G.get()
    X_pred = np.array([age, gender, TB, DB, alkphos, sgpt, sgot, TP, ALB, A_G], dtype=float).reshape(1, -1)
    y_svm = model.predict(X_pred)
    messagebox.showinfo("Kết quả dự đoán theo svm",
                        "Người bị bệnh gan (1 là bị nhiễm) , (2 là không bị ): " + str(y_svm))


def perdict_logistic():
    y_logitic = logisticRegression.predict(X_test)
    messagebox.showinfo("Khả năng dự đoán của LogisticRegression ",
                        "Độ chính xác của phương pháp Logitic Regression: " + str(
                            accuracy_score(y_test, y_logitic) * 100) + "%")


def perdict_svm():
    y_svm = model.predict(X_test)
    messagebox.showinfo("Khả năng dự đoán của SVM ",
                        "Độ chính xác của phương pháp SVM : " + str(accuracy_score(y_test, y_svm) * 100) + "%")


btn_logistic = Button(form, text='Kết quả dự đoán logisticRegression', command=logistic)
btn_logistic.grid(row=12, column=2, pady=20)

btn_svm = Button(form, text='Kết quả dự đoán svm', command=svm)
btn_svm.grid(row=13, column=2, pady=20)

btn_logistic1 = Button(form, text='Khả năng dự đoán logisicRegression', command=perdict_logistic)
btn_logistic1.grid(row=12, column=3, padx=30)

btn_logistic2 = Button(form, text='Khả năng dự đoán svm', command=perdict_svm)
btn_logistic2.grid(row=13, column=3, padx=30)

form.mainloop()