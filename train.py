import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,classification_report
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib


#Load_data

df=pd.read_csv("data/loan.csv")
df.columns = df.columns.str.strip()

# Strip spaces from all string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

print(df.head())
print(df.isnull().sum())

#Preprocessing

df=df.dropna()
label_encoders={}

for col in df.select_dtypes(include="object").columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    label_encoders[col]=le

# Save label encoders for use in app.py
joblib.dump(label_encoders, "model/label_encoders.pkl")
print("Label encoders saved!")
print("Encoders:", label_encoders)

#Split_data

x=df.drop("loan_status",axis=1)
y=df["loan_status"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#01. Train classification model

clf=LGBMClassifier()
clf.fit(x_train,y_train)

#predictions

y_pred=clf.predict(x_test)

#Evaluation

print("Accuracy: ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#02. Risk score model

reg=LGBMRegressor()
reg.fit(x_train,y_train)

#Predict risk score
y_pred_reg=reg.predict(x_test)

#convert to error metric

rmse=np.sqrt(mean_squared_error(y_test,y_pred_reg))

print("RMSE:", rmse)

#Save classification model(Loan Approval)

joblib.dump(clf, "model/model_classifier.pkl")

#Save regression model(Risk Score)

joblib.dump(reg, "model/model_regressor.pkl")
