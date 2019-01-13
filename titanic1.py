# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 22:12:20 2018

@author: archit bansal
"""
#ipmorting the libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
#importing the data
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

#splitting the dependent and independent variables
train_x=train_data.drop(["Survived"],axis=1)
train_y=train_data["Survived"]

#dropping the unnecessary variables
train_x=train_x.drop(['Name','Ticket','Cabin'],axis=1)
test_data=test_data.drop(['Name','Ticket','Cabin'],axis=1)


#analyzing the features
print(train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False))
print(train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False))
print(train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False))
print(train_data[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False))



#taking care of missing values
#to use imputer we have convert dataframe to ndarray as it uses slice metohd which is the part of numpy array
'''from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values=float('NaN'),strategy='mean',axis=0)
imputer = imputer.fit(train_x[:,[4]])
train_x[:,[4]] = imputer.transform(train_x[:,[4]])'''
train_x["Age"]=train_x["Age"].fillna(train_x["Age"].mean())
#we have replace it by S as the average of S is maximum in Embarked 
train_x["Embarked"]=train_x["Embarked"].fillna("S")
#for test set
test_data["Age"]=test_data["Age"].fillna(test_data["Age"].mean())
test_data["Embarked"]=test_data["Embarked"].fillna("S")



#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

#converting in form of ndarray as we are using slicing
train_X=train_x.values
train_X[:,2]=le.fit_transform(train_X[:,2])
train_X[:,7]=le.fit_transform(train_X[:,7])
train_x=pd.DataFrame(train_X,columns=train_x.columns)
test_X=test_data.values
test_X[:,2]=le.fit_transform(test_X[:,2])
test_X[:,7]=le.fit_transform(test_X[:,7])
test_data=pd.DataFrame(test_X,columns=test_data.columns)



#one hot encoding
'''onehotencoder = OneHotEncoder()
train_x1= onehotencoder.fit_transform(train_x).toarray()'''
train_x=pd.get_dummies(train_x,columns=["Sex","Embarked"])
test_data=pd.get_dummies(test_data,columns=["Sex","Embarked"])
#avoid dummy variable trap
train_x=train_x.drop(["Embarked_2"],axis=1)
test_data=test_data.drop(["Embarked_2"],axis=1)

#converting narray to dataframe
'''#converting in form of ndarray
train_X=train_x.values
train_x=pd.DataFrame(train_x,columns=['Passengerid','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])'''
train_x.info()
train_x.describe()
train_x.describe(include=["O"])
test_data.describe(include=["O"])
test_data.describe()
x_train=train_x.drop(["PassengerId"],axis=1)
x_test=test_data.drop(["PassengerId"],axis=1).copy()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
#it will forward fill the non-nan values with nan values
x_test = x_test.fillna(method='ffill')
x_test = sc_X.transform(x_test)
#training the model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=80,criterion="entropy",random_state=0)
classifier.fit(x_train,train_y)
#calculating accuracy
acc_random_forest = round(classifier.score(x_train,train_y) * 100, 2)
print(acc_random_forest)
# Predicting the Test set results
pred_y = classifier.predict(x_test)
y=pd.read_csv("gender_submission.csv")
y=y.iloc[:,1]
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,pred_y)
'''
#using SVM
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', random_state = 0)
classifier1.fit(x_train,train_y)

# Predicting the Test set results
pred_y_svm = classifier.predict(test)
print(classifier1.score(x_train,train_y))
'''
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived":pred_y
    })
submission.to_csv('titanic1', index=False)


