# HERE WE SHALL APPEND MODEL 1 , BASED ON LP 5
'''
LP5 MODEL IS A CLASIFICATION PROBLEM THE OUTPUT BEING POSITIVE OR NEGATIVE 
    We will use a label encoder on the target column so as to restrict the 
    output to 2 outputs 
    1 - Churn Positive
    0 - Churn Negative

PREDICTING 
     Churn              object

FEATURES / LABELS 
     gender             object
     SeniorCitizen      int64
     Partner            object
     Dependents         object
     tenure             int64  
     PhoneService       object 
     MultipleLines      object 
     InternetService    object 
     OnlineSecurity     object 
     OnlineBackup       object 
     DeviceProtection   object 
     TechSupport        object 
     StreamingTV        object 
     StreamingMovies    object 
     Contract           object 
     PaperlessBilling   object 
     PaymentMethod      object 
     MonthlyCharges     float64
     TotalCharges       object 
 

'''

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import load
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# Load your dataframe
df = pd.read_csv('C:\Users\user\Desktop\FAST_API\src\Telco-Customer-Churn.csv')

# 1. Drop specified columns
drop_columns = ['customerID', 'gender', 'PaymentMethod', 'StreamingMovies']
df = df.drop(drop_columns, axis=1)

# 2. Convert TotalCharges columns to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Scale numeric columns
numeric_features = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']
numeric_transformer = StandardScaler()

# 4. Encode categorical columns
categorical_features = ['Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'Contract', 'PaperlessBilling']
categorical_transformer = OneHotEncoder(sparse=False)

# 5. Encode the target column
target_column = 'Churn'
target_encoder = LabelEncoder()

# 6. Split into X and y
X = df.drop(target_column, axis=1)                       #X = df.drop("Churn", asix = 1 )
y = target_encoder.fit_transform(df[target_column])

# 7. Create the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 8. Split data

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier())
])

# 9. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Fit the pipeline and make predictions
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# CURATE THE APP NOW WITH THE MODEL BEING THE PIPELINE 
from typing import Union
from fastapi import FastAPI
app = FastAPI()
#[SeniorCitizen, Partner, Dependents, tenure , PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,DeviceProtection, TechSupport
#,StreamingTV, Contract, PaperlessBilling ,MonthlyCharges ,TotalCharges]
description_1 = ['''
                LP5 MODEL IS A CLASIFICATION PROBLEM THE OUTPUT BEING POSITIVE OR NEGATIVE. 
                This is a Churn Analysis problem where we analysed Vodaphone company customer details and 
                Come up with a ML model to be able to predict a customers' likelihood of churning out of using the company's products and services. 

                The columns provided in the dataset are as follows 
                gender             object
                SeniorCitizen      int64
                Partner            object
                Dependents         object
                tenure             int64  
                PhoneService       object 
                MultipleLines      object 
                InternetService    object 
                OnlineSecurity     object 
                OnlineBackup       object 
                DeviceProtection   object 
                TechSupport        object 
                StreamingTV        object 
                StreamingMovies    object 
                Contract           object 
                PaperlessBilling   object 
                PaymentMethod      object 
                MonthlyCharges     float64
                TotalCharges       object 

        The Model created has an accuracy of .....  Feel free to interact with our model here. ''']

@app.post("/predictions churn analysis", description= description_1 )
async def prediction_churn (SeniorCitizen, Partner, Dependents, tenure , PhoneService, 
                            MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
                            DeviceProtection, TechSupport,StreamingTV, Contract, 
                            PaperlessBilling ,  MonthlyCharges ,TotalCharges):
    
    input_ = {"SeniorCitizen" : SeniorCitizen, "Partner" : Partner ,"Dependents" : Dependents, "tenure": tenure ,
              "PhoneService": PhoneService, "MultipleLines" : MultipleLines, "InternetService" :InternetService,
              "OnlineSecurity":OnlineSecurity, "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
             "TechSupport" : TechSupport, "StreamingTV" : StreamingTV, "Contract": Contract, "PaperlessBilling": PaperlessBilling ,
               "MonthlyCharges":  MonthlyCharges ,"TotalCharges":TotalCharges

    }

    input_df = pd.DataFrame([input_])

    # make the predictions
    pred = pipeline.predict(input_df)

    return {"predictions on Churn dataset": pred.tolist()}




    