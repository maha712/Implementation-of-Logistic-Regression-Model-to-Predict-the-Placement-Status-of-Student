# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values.

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/

import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#removes the specified row or column
data1.head()

data1.isnull().sum() #to check for null values

data1.duplicated().sum() #to check for duplicate values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)#accuracy score=(TP+TN)/(TP+FN+TN+FP),True+ve/
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:

1.Placement data
![placement data](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/be343736-f12f-4b1d-9ada-49e804abacb6)

2.Salary Data
![salary data](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/c9f9520c-3761-428a-a1b6-f2253b57bbb9)

3.Checking the null() function
![checking the null() function](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/e9ac4675-3626-46cc-bda4-1af731d9a3e5)

4.Data duplicate
![data duplicating](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/36b1ed15-ddbe-4612-b934-e497a7df1580)

5.print Data
![print data](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/52eae1ff-ef0e-4dcb-aa16-659b66a1279f)

6.Data status
![data status](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/40f04c8d-202a-44dd-8e78-ccd34e7e5c45)

7.y_prediction array
![y prediction array](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/db42a52b-32ba-44c0-98dd-471267daab68)

8.Accuracy value
![accuracy value](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/7838376f-21bf-4adb-bcb8-97dfeaa571a2)

9.Confusion array
![confusion array](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/26db49bc-c82c-495d-8394-3abbf736606c)

10.Classification report
![classification report](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/e518e27b-32c8-495a-b462-f8048dbabe25)

11.prediction of LR
![prediction of LR](https://github.com/maha712/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121156360/42858a2e-fab1-4faf-90f4-0230ae92a5a8)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
