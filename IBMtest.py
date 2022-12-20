# Attrition Ex
"""

# Description : This program predicts employee attrition

# Import the Libraries
import numpy as np
import pandas as pd
import seaborn as sns

# Load the data
from google.colab import files
uploaded = files.upload()

#Store the data into a dataframe
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
#print first 7
df.head(7)

#Get the number fo rows and cols
df.shape

#Get the column data tptes
df.dtypes

#Get a count of the empty values for each column
df.isna().sum()

#Check for any missing or null vals
df.isnull().values.any()

#View some statistics
df.describe()

#Get a count of the number of employees that stayed and left the company
df['Attrition'].value_counts()

#Visualise the number of employees that stayed and left the company
sns.countplot(df['Attrition'])

#Always guessing no accuracy percentage of attrition
(1233-237)/1233

#Show the number of employees that left and stayed by their age
import matplotlib.pyplot as plt
plt.subplots(figsize=(12,4))
sns.countplot(x='Age',hue='Attrition',data=df, palette='colorblind')

#Print all of the data types and their unique values
for column in df.columns:
  if df[column].dtype == object:
    print(str(column) + ' : '+ str(df[column].unique()))
    print(df[column].value_counts())
    print('______________________________________________________')

df['StandardHours'].unique()

#Remove some useless columns
df = df.drop('Over18', axis = 1) #all yes
df = df.drop('EmployeeNumber', axis = 1) #id
df = df.drop('StandardHours', axis = 1) #80 unique vals
df = df.drop('EmployeeCount', axis = 1) #1 val

#Get the correlation
df.corr()

#Visualise the correlation
plt.figure(figsize=(14,14))
sns.heatmap(df.corr(), annot=True, fmt='.0%')

#Transform the data
#Transform non-numerical into numerical
from sklearn.preprocessing import LabelEncoder

for column in df.columns:
  if df[column].dtype == np.number:
    continue
  df[column] = LabelEncoder().fit_transform(df[column])

#Create a new column, put attrition first
df['Age_Years'] = df['Age']
df = df.drop('Age',axis=1)

#Split the data
X = df.iloc[:, 1:df.shape[1]].values
Y = df.iloc[:, 0].values

#Split the data into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Use the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)

#Get the accuracy on the training data set
forest.score(X_train, Y_train)

#Show the confusion matrix and accuracy score for the model on the test data
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, forest.predict(X_test))

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print(cm)
print('Model Testing Accuracy = {}'.format( (TP + TN)/(TP +TN + FN + FP) ))

importances = pd.DataFrame({'feature':df.iloc[:,1:df.shape[1]].columns, 'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances

#Visualise the importance
importances.plot.bar()
