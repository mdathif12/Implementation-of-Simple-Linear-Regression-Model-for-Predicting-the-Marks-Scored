# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: mohamed athif rahuman J
RegisterNumber:212223220058  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset
![image](https://github.com/mdathif12/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365313/f3ee5fe4-b514-4f11-bc95-2242354b644e)
Head Values
![image](https://github.com/mdathif12/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365313/1da6c09f-2074-486b-85cf-abda8cfe93f8)
Tail Values
![image](https://github.com/mdathif12/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365313/d497894c-4640-4e43-b243-111dbdd10155)
X and Y values
![image](https://github.com/mdathif12/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365313/07eaa7d4-32e7-45a7-b1bf-f098ad35b3cf)
Predication values of X and Y
![image](https://github.com/mdathif12/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365313/cea39ad5-3e47-4dd0-a692-d107832030c6)
MSE,MAE and RMSE
![image](https://github.com/mdathif12/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365313/d134d2b1-d2b0-4d9c-ad7f-60240364de48)
Training Set
![image](https://github.com/mdathif12/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365313/742310cf-967c-4ac5-804b-999a0265ca83)
Testing Set
![image](https://github.com/mdathif12/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365313/788e349b-f618-4783-85b1-92fa0ede6c6b)


Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
