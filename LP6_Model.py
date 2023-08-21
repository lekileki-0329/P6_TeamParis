# HERE WE SHALL APPEND MODEL 2 , BASED ON LP 6
"""
LP6 MODEL IS A CLASIFICATION PROBLEM THE OUTPUT BEING POSITIVE OR NEGATIVE 
    We will use a label encoder on the target column so as to restrict the 
    output to 2 outputs 
    1 - Sepssis Positive
    0 - Sepssis Negative

PREDICTING 
    Sepssis 

FEATURES / LABELS 
    'PRG' : 'PlasmaGlucose',
    'PL' : 'Blood_W_Result_1',
    'PR' : 'BloodPressure',
    'SK' : 'Blood_W_Result_2',
    'TS' : 'Blood_W_Result_3',
    'M11' : 'BMI',
    'BD2' : 'Blood_W_Result_4',
    'Age' : 'Age',
    'Insurance' : 'Insurance'

"""

import pandas as pd
import numpy as np

# Importing Encoders and scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Importing splitting module
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

# Create an instance of the trainer
gradient = GradientBoostingClassifier()

# While reading in local machine
train = pd.read_csv("C:\\Users\\user\\Downloads\\P6\\Paitients_Files_Train.csv")
eval = pd.read_csv("C:\\Users\\user\\Downloads\\P6\\Paitients_Files_Test.csv")

col_name_1 = {
    "ID": "ID",
    "PRG": "PlasmaGlucose",
    "PL": "Blood.W.Result-1",
    "PR": "BloodPressure",
    "SK": "Blood.W.Result-2",
    "TS": "Blood.W.Result-3",
    "M11": "BMI",
    "BD2": "Blood.W.Result-4",
    "Age": "Age",
    "Insurance": "Insurance",
}

train.rename(columns=col_name_1, inplace=True)

eval.rename(columns=col_name_1, inplace=True)

train.drop("ID", axis=True, inplace=True)

scaler = StandardScaler()
X = train.drop("Sepssis", axis=1)
y = train["Sepssis"]

X_scaled = scaler.fit_transform(X)
X_scaled_ = pd.DataFrame(X_scaled)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded_ = pd.DataFrame(y_encoded)

train_ready = pd.concat([X_scaled_, y_encoded_], axis=1)


# Select features using slicing
X_train = train_ready.iloc[:, :-1]
# Selecting the target using target
y_train = pd.DataFrame(train_ready.iloc[:, -1])

x_train, x_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)  # stratify = y_train)

# Train
gradient.fit(x_train, y_train)

# save in a variable

model = gradient.fit(x_train, y_train)

# pred = model.predict(customer_input)

# return{'output': "pred"}
# Make predictions
"""
pred = (gradient.fit(x_train, y_train)).predict(x_test)
model.predict(input_)
where the X_test in this case will be our input from the user 

"""
# CURATE THE MODEL WITH model being the model instatiated
# use a get request

from typing import Union
from fastapi import FastAPI

app = FastAPI()


# pred = model.predict(input_)
from joblib import load

# Load your trained machine learning model
# model = joblib.load("path_to_your_model.pkl")
description_1 = "LP5 MODEL: Churn Analysis Classification\n\ The LP5 model is designed to address a classification problem where the output is either positive or negative. This project focuses on Churn Analysis, specifically analyzing customer details from Vodafone company. The goal is to develop an ML model capable of predicting a customers likelihood of churning – discontinuing the use of the company's products and services.\n\n Dataset Columns:\n"
description_2 = [
    "- gender: object",
    "- SeniorCitizen: int64",
    "- Partner: object",
    "- Dependents: object",
    "- tenure: int64",
    "- PhoneService: object",
    "- MultipleLines: object",
    "- InternetService: object",
    "- OnlineSecurity: object",
    "- OnlineBackup: object",
    "- DeviceProtection: object",
    "- TechSupport: object",
    "- StreamingTV: object",
    "- StreamingMovies: object",
    "- Contract: object",
    "- PaperlessBilling: object",
    "- PaymentMethod: object",
    "- MonthlyCharges: float64",
    "- TotalCharges: object",
]
other = [
    "The accuracy of the created model is yet to be determined.\n\n",
    "Feel free to interact with our model here!",
]

# final = description_1 + description_2 + other
final = "LP5 MODEL: Churn Analysis Classification\n\ The LP5 model is designed to address a classification problem where the output is either positive or negative. This project focuses on Churn Analysis, specifically analyzing customer details from Vodafone company. The goal is to develop an ML model capable of predicting a customers likelihood of churning – discontinuing the use of the company's products and services.\n\n Dataset Columnsgender: object- SeniorCitizen: int64,- Partner: object,- Dependents: object,- tenure: int64,- PhoneService: object,- MultipleLines: object,- InternetService: object,- OnlineSecurity: object,- OnlineBackup: object,- DeviceProtection: object,- TechSupport: object,- StreamingTV: object,- StreamingMovies: object,- Contract: object,- PaperlessBilling: object,- PaymentMethod: object,- MonthlyCharges: float64,- TotalCharges: object"


@app.post("/predictions sepsiss", description=final)
async def prediction(
    PlasmaGlucose: float,
    Blood_W_Result_1: float,
    BloodPressure: float,
    Blood_W_Result_2: float,
    Blood_W_Result_3: float,
    BMI: float,
    Blood_W_Result_4: float,
    Age: int,
    Insurance: int,
):
    input_data = {
        "PlasmaGlucose": PlasmaGlucose,
        "Blood_W_Result_1": Blood_W_Result_1,
        "BloodPressure": BloodPressure,
        "Blood_W_Result_2": Blood_W_Result_2,
        "Blood_W_Result_3": Blood_W_Result_3,
        "BMI": BMI,
        "Blood_W_Result_4": Blood_W_Result_4,
        "Age": Age,
        "Insurance": Insurance,
    }

    # pass input into a DataFrame

    input_df = pd.DataFrame([input_data])

    # make predictions
    pred = model.predict(input_df)

    # We have an output of either 1 or 0 since our target was encoded
    def analyze_sepsis_prediction(pred):
        if pred == [1]:
            print("Sepsis Positive")
        else:
            print("Sepsis Negative")

    # Creating a dictionary with a message about the prediction
    return {"predictions on sepsis dataset": f"The patient is most likely to be {pred}"}


# NOW WE WANT TO LOOK AT THE EDA TRY TO APPEND IT USING THE .GET REQUEST
# @app.get("/EDA")
# async def eda_function(
#     PlasmaGlucose: float,
#     Blood_W_Result_1: float,
#     BloodPressure: float,
#     Blood_W_Result_2: float,
#     Blood_W_Result_3: float,
#     BMI: float,
#     Blood_W_Result_4: float,
#     Age: int,
#     Insurance: int,
# ):
#     input_data = {
#         "PlasmaGlucose": PlasmaGlucose,
#         "Blood_W_Result_1": Blood_W_Result_1,
#         "BloodPressure": BloodPressure,
#         "Blood_W_Result_2": Blood_W_Result_2,
#         "Blood_W_Result_3": Blood_W_Result_3,
#         "BMI": BMI,
#         "Blood_W_Result_4": Blood_W_Result_4,
#         "Age": Age,
#         "Insurance": Insurance,
#     }
#     # make predictions
#     pred = model.predict(input_data)

#     # pass input into a DataFrame
#     input_ = pd.DataFrame([input_data].append({"sepsis": pred}))

#     # Make a correlation matrix

#     # calculate correlation matrix
#     correlation_matrix = input_.corr()

#     return {
#         "input_df": input_.to_dict(),
#         "correlation_matrix": correlation_matrix.to_dict(),
#     }
