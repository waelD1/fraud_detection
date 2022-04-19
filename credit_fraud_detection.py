# data manipulation packages
import pandas as pd
import numpy as np
import pickle


#ML libraries

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

#Metrics Libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix


#Graphic Libraries
import plotly.express as px


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
# For the training set
transformer = RobustScaler().fit(X_train)
scaled_features_train = transformer.transform(X_train)
scaled_features_train_df = pd.DataFrame(scaled_features_train, index=X_train.index, columns=X_train.columns)

# For the test set
transformer = RobustScaler().fit(X_test)
scaled_features_test = transformer.transform(X_test)
scaled_features_test_df= pd.DataFrame(scaled_features_test, index=X_test.index, columns=X_test.columns)

# Saving the scaler
# with open('scaler_model.pkl', 'wb') as f:
#   pickle.dump(transformer_train, f)


#creating the objects
logreg_cv = LogisticRegression(solver='liblinear',random_state=123)
dt_cv=DecisionTreeClassifier(random_state=123)
knn_cv=KNeighborsClassifier()
#svc_cv=SVC(kernel='linear',random_state=123)
nb_cv=GaussianNB()
rf_cv=RandomForestClassifier(random_state=123)
cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree',2:'KNN',3:'Naive Bayes',4:'Random Forest'}
cv_models=[logreg_cv,dt_cv,knn_cv,nb_cv,rf_cv] # svc_cv


for i,model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, scaled_features_train_df, y_train, cv=10, scoring ='f1_weighted').mean()))



#Predict with the selected best parameter
#y_pred= scaled_features_test_df
rf=RandomForestClassifier(random_state=123)
rf.fit(scaled_features_train_df, y_train)

y_pred= rf.predict(scaled_features_test_df)

#Plotting confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(rf, scaled_features_test_df, y_test)



#Classification metrics
print(classification_report(y_test, y_pred, target_names=['Not Fraud','Fraud']))

# The result are quit good

# saving the model
#save the word index
# with open('credit_fraud_model.pkl', 'wb') as f:
#     pickle.dump(rf, f)

#load the word index
# with open('credit_fraud_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)
