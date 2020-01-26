# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
df = pd.read_csv(path)
# print(df.head())
X = df.iloc[:,1:-1]
y = df['Churn']
# print(X.head())
# print(y.head())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=0)



# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train['TotalCharges'] = X_train['TotalCharges'].replace(' ',np.NaN)
# print(X_train['TotalCharges'])
X_test['TotalCharges'] = X_test['TotalCharges'].replace(' ',np.NaN)
# print(X_test['TotalCharges'])
X_train['TotalCharges'] = X_train['TotalCharges'].astype('float64')
# print(dtype(X_train['TotalCharges']))
print(X_train.info())
X_test['TotalCharges'] = X_test['TotalCharges'].astype('float64')
X_train['TotalCharges'] = X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean())
# print(X_test['TotalCharges'])
X_test['TotalCharges'] = X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean())
# print(X_train.isnull().sum(),X_test.isnull().sum())
cat_cols = X_train.select_dtypes(include='object').columns
# print(cat_cols)
le = LabelEncoder()
for col in cat_cols:
    for q in cat_cols:
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

y_train = y_train.replace({'No':0,'Yes':1})
y_test = y_test.replace({'No':0,'Yes':1})




# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
# print(X_train.head(),'\n',X_test.head(),'\n',y_train.head(),'\n',y_test.head())
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred = ada_model.predict(X_test)
ada_score = accuracy_score(y_test,y_pred)
print(ada_score)
ada_cm  = confusion_matrix(y_test,y_pred)
print(ada_cm)
ada_cr = classification_report(y_test,y_pred)
print(ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_test,y_pred)
print(xgb_score)
xgb_cm = confusion_matrix(y_test,y_pred)
print(xgb_cm)
xgb_cr = classification_report(y_test,y_pred)
print(xgb_cr)

clf_model = GridSearchCV(estimator=xgb_model,param_grid=parameters)
clf_model.fit(X_train,y_train)
y_pred = clf_model.predict(X_test)
clf_score = accuracy_score(y_test,y_pred)
print(clf_score)
clf_cm = confusion_matrix(y_test,y_pred)
print(clf_cm)
clf_cr = classification_report(y_test,y_pred)
print(clf_cr)
# print(clf_model.bes)



