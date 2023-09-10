#!/usr/bin/env python

# H1V1 Vaccine Prediction Using Survey Data - Himanshu Raj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, classification_report, confusion_matrix

# Load the dataset from the URL
url = 'https://raw.githubusercontent.com/Premalatha-success/Datasets/main/h1n1_vaccine_prediction.csv'
data = pd.read_csv(url)
print(data.shape)

# Define features and target variable
X = data.drop('h1n1_vaccine', axis=1)  # Features
y = data['h1n1_vaccine']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define categorical and numerical columns
categorical_cols = ['age_bracket', 'qualification', 'race', 'sex', 'income_level', 'marital_status', 'housing_status', 'employment', 'census_msa']
numerical_cols = ['h1n1_worry', 'h1n1_awareness', 'antiviral_medication', 'contact_avoidance', 'bought_face_mask', 'wash_hands_frequently', 'avoid_large_gatherings', 'reduced_outside_home_cont', 'avoid_touch_face', 'dr_recc_h1n1_vacc', 'dr_recc_seasonal_vacc', 'chronic_medic_condition', 'cont_child_undr_6_mnths', 'is_health_worker', 'has_health_insur', 'is_h1n1_vacc_effective', 'is_h1n1_risky', 'sick_from_h1n1_vacc', 'is_seas_vacc_effective', 'is_seas_risky', 'sick_from_seas_vacc', 'no_of_adults', 'no_of_children']


# Preprocessing: Handle missing values and perform one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])


# Fill missing values in numerical columns with the mean
X_train.loc[:, numerical_cols] = X_train.loc[:, numerical_cols].fillna(X_train[numerical_cols].mean())
X_test.loc[:, numerical_cols] = X_test.loc[:, numerical_cols].fillna(X_train[numerical_cols].mean())


# Create pipelines for each classifier
logistic_regression = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

random_forest = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

knn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # Adjust n_neighbors as needed
])

# Fit and evaluate each model
models = {'Logistic Regression': logistic_regression, 'Random Forest': random_forest, 'KNN': knn}

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f'Model: {name}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')
    print('\n') 

    # ROC Curve
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
#     plt.show()
print('\nDONE')


'''
Model: Logistic Regression
Accuracy: 0.8386
F1 Score: 0.5276
ROC-AUC: 0.6877


Model: Random Forest   ## BEST with F1 Score of 0.54 and Accuracy of 84.9%
Accuracy: 0.8491
F1 Score: 0.5467
ROC-AUC: 0.6959


Model: KNN
Accuracy: 0.8109
F1 Score: 0.4752
ROC-AUC: 0.6624
'''

