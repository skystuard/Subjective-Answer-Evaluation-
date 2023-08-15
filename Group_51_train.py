import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib

def train():
    data = pd.read_csv('Group_51_data/scoredata.csv')
    X = data.iloc[:, 0:3].values
    Y = data.iloc[:, 3].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=80)
    regressor = RandomForestClassifier(n_estimators=30, random_state=0)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    print(y_pred)
    print(y_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    
    
    joblib.dump(regressor, "Group_51_models/custom_trained_random_forest1.joblib")

train()
