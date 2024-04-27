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
Developed by: Mohamed Athif Rahuman J
RegisterNumber:  212223220058
*/
```
```import pandas as pd
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
## dataset
![dataset](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/14a89059-3d77-4d08-a327-666017ba648b)

## head values
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/4eea6a9b-8a1a-4f07-b592-578af9337f5b)

## tail values
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/e9ca07f3-a57c-4c8e-8870-ddb2e868823b)
## X and Y values
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/c5aceffc-a472-4054-acb3-8e6e0dd02116)

## Predication values of X and Y
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/2c7a26b6-51b8-4a9a-8f94-96e1ed2aafcd)
## MSE,MAE and RMSE
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/7ed57fce-a2a7-4784-8f04-65190ec687d6)

## Training Set
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/e3fe1065-e497-440d-8fc2-d86e8d98181c)

## Test Set Graph
![image](https://github.com/Wkrish28/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144295230/5d9f0ced-bd0f-4c10-88d6-4fbeb3938db7)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
