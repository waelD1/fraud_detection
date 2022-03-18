# import packages
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.express as px
import random

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

#Metrics Libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



df = pd.read_csv('credit_fraud_detection.csv', 
                 header = 0
                 )

### Resampling the dataset

# Class count
count_class_0, count_class_1 = df['isFraud'].value_counts()

# Divide by class
df_class_0 = df[df['isFraud'] == 0]
df_class_1 = df[df['isFraud'] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df_resample = pd.concat([df_class_0_under, df_class_1], axis=0)
print('Random under-sampling:')
print(df_resample.isFraud.value_counts())

df_resample.isFraud.value_counts().plot(kind='bar', title='Count (target)');

df_resample.to_pickle("./df_resample.pkl")  


#read the pickled dataframe
df = pd.read_pickle("./df_resample.pkl")  

#type of transaction when it is fraudulent
fig = px.box(df, x="type", y="amount", color = 'isFraud')
fig.show()
#All the fraudulent transactions are in 'CASH_OUT' and 'TRANSFER' type.


# Look at the rate of change between account
mean_rate_of_change_no_fraud = ((round(df[(df['isFraud'] == 0)]['newbalanceOrig'].mean(), 1) - round(df[(df['isFraud'] == 0)]['oldbalanceOrg'].mean(),1)) / round(df[(df['isFraud'] == 0)]['oldbalanceOrg'].mean(),1))*100
mean_rate_of_change_fraud = ((round(df[(df['isFraud'] == 1)]['newbalanceOrig'].mean(), 1) - round(df[(df['isFraud'] == 1)]['oldbalanceOrg'].mean(),1)) / round(df[(df['isFraud'] == 1)]['oldbalanceOrg'].mean(),1))*100

print('rate of change between new balance and old balance for non fraud cases : ', round(mean_rate_of_change_no_fraud, 2),  '%')
print('rate of change between new balance and old balance for fraudulent cases : ', round(mean_rate_of_change_fraud, 2), '%')
#rate of evolution is huge in case of fraud, very small when it is not the case. So the fraud happends with huge variation between the old amount and the new amount

mean_rate_of_change_no_fraud_dest = ((round(df[(df['isFraud'] == 0)]['newbalanceDest'].mean(), 1) - round(df[(df['isFraud'] == 0)]['oldbalanceDest'].mean(),1)) / round(df[(df['isFraud'] == 0)]['oldbalanceDest'].mean(),1))*100
mean_rate_of_change_fraud_dest = ((round(df[(df['isFraud'] == 1)]['newbalanceDest'].mean(), 1) - round(df[(df['isFraud'] == 1)]['oldbalanceDest'].mean(),1)) / round(df[(df['isFraud'] == 1)]['oldbalanceDest'].mean(),1))*100

print('rate of change between new balance and old balance for non fraud cases : ', round(mean_rate_of_change_no_fraud_dest, 2),  '%')
print('rate of change between new balance and old balance for fraudulent cases : ', round(mean_rate_of_change_fraud_dest, 2), '%')

# Same thing on the side of the one who receives the money
# These variables are interesting, we can create features to reflect the rates of change between accounts

# variables rate of change
##Sender's rate of change
df['mean_rate_of_change_orig'] = np.where(
                              ((df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0) )|
                              (df['oldbalanceOrg'] == 0),
                              0,
                              round(((df['newbalanceOrig'] - df['oldbalanceOrg'])/df['oldbalanceOrg']),2) 
                              )
## Receiver's balance
df['mean_rate_of_change_dest'] = np.where(
                                      ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)) |
                                      (df['oldbalanceDest'] == 0),
                                        0, 
                                        round(((df['newbalanceDest'] - df['oldbalanceDest'])/df['newbalanceDest']),2)
                                        )

# check the infinite value
df[df['mean_rate_of_change_dest'] == -np.inf]['newbalanceDest'].unique()

# If newbalanceDest is equal to 0 then the value is equal to -np.inf
# We replace the -inf values with -1000
df['mean_rate_of_change_dest'] = np.where(df['mean_rate_of_change_dest'] == -np.inf, -1000, df['mean_rate_of_change_dest'])

# Preprocessing
## Feature Encoding and concatenating the results
new_df = pd.concat([df, pd.get_dummies(df['type'], prefix = 'type_payment')], axis = 1)

#Dropping useless columns
new_df.drop(columns = ['type', 'nameOrig', 'nameDest', 'isFlaggedFraud'], inplace = True)

#splitting the dataset
from sklearn.model_selection import train_test_split

X = new_df.loc[:, new_df.columns != 'isFraud']
y = new_df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Creating Scaled features
from sklearn.preprocessing import RobustScaler

# For the training set
transformer = RobustScaler().fit(X_train)
scaled_features_train = transformer.transform(X_train)
scaled_features_train_df = pd.DataFrame(scaled_features_train, index=X_train.index, columns=X_train.columns)

# For the test set
transformer = RobustScaler().fit(X_test)
scaled_features_test = transformer.transform(X_test)
scaled_features_test_df= pd.DataFrame(scaled_features_test, index=X_test.index, columns=X_test.columns)