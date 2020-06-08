
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

#Reading data set
df = pd.read_csv('breast-cancer.csv')


#Here we are dropping 2 columns which are ineffective for classification
df.drop("id",axis=1,inplace=True)
df.drop("Unnamed: 32",axis=1,inplace=True)

sns.countplot(df['diagnosis'],label="Count", palette="Set2")
#In this data set number of benign is more than malignant, so we can work on it


whole = list(df.columns[1:])

#Now we transform our categorical output into integer value.

df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

#variables
prediction_var = whole

#Creating train and test sets 
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3) #0.7 train 0.3 test

train_X = train[prediction_var]
train_y=train.diagnosis
test_X= test[prediction_var] 
test_y =test.diagnosis   



#Random Forest Model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100) # creating random forest model named model
model.fit(train_X,train_y) # we put our train sets in it.
prediction=model.predict(test_X) # prediction of test data (variables)
print("model 1 =",metrics.accuracy_score(prediction,test_y) )# Then we see accuracy results in here


#SVM (Support Vector Machine)
from sklearn import svm
model2 = svm.SVC(gamma='auto')
model2.fit(train_X,train_y)
prediction2=model2.predict(test_X)
print("model 2 =",metrics.accuracy_score(prediction2,test_y))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(train_X,train_y)
prediction3 = model3.predict(test_X)
print("model 3 =",metrics.accuracy_score(prediction3,test_y))

#K-NN
from sklearn.neighbors import KNeighborsClassifier
model4 = KNeighborsClassifier()
model4.fit(train_X,train_y)
prediction4 = model4.predict(test_X)
print("model 4 =",metrics.accuracy_score(prediction4,test_y))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model5 = LogisticRegression(random_state=10, solver='lbfgs')
model5.fit(train_X,train_y)
prediction5 = model5.predict(test_X)
print("model 5 =",metrics.accuracy_score(prediction5,test_y))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
model6 = GaussianNB()
model6.fit(train_X,train_y)
prediction6 = model6.predict(test_X)
print("model 6 =",metrics.accuracy_score(prediction6,test_y))
