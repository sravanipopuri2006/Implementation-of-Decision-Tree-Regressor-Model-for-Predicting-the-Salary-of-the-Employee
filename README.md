# EXP7-Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2. 


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: POPURI SRAVANI
RegisterNumber: 212223240117 
*/
import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print(mse)

r2=metrics.r2_score(y_test,y_pred)
print(r2)

dt.predict([[5,6]])

```

## Output:
## DATASET
![Screenshot 2024-04-02 132510](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/139778301/daeaeaea-0064-4533-ad2c-95b46f981a78)
## data.info()
![Screenshot 2024-04-02 132522](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/139778301/a0eb11a7-ce65-4dc2-b5c5-c4b23069259b)
## Checking if null values are present
![Screenshot 2024-04-02 132533](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/139778301/fcef250f-d622-4e61-8ca6-452fcf7dd60f)
## Dataset after encoding
![Screenshot 2024-04-02 132543](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/139778301/8b78ab1c-8bc9-4f90-94bc-1f07e0179ca1)

## MSE
![Screenshot 2024-04-02 132553](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/139778301/7d9b7e2f-ec25-4cbf-86ad-fb8f8bdfb6f4)
## r2
![Screenshot 2024-04-02 132606](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/139778301/9c1627ae-b5bb-4b55-becf-0589ad46be26)

## dt.predict()
![Screenshot 2024-04-02 132616](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/139778301/331b40b1-77c1-444f-ac09-fbd4ed1eded2)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
