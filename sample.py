import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


age = 1
blood_pressure = 1
heart_rate = 1
respiration = 1

DATA_CSV_FILE = pd.read_csv('ride_data_set.csv')
DATA_CSV_FILE.isnull().sum()

print(DATA_CSV_FILE)
X = pd.DataFrame(np.c_[
    DATA_CSV_FILE['min_age'],
    DATA_CSV_FILE['max_age'],
    DATA_CSV_FILE['low_systolic'],
    DATA_CSV_FILE['low_diastolic'],
    DATA_CSV_FILE['high_systolic'],
    DATA_CSV_FILE['high_diastolic'],
    DATA_CSV_FILE['heart_below'],
    DATA_CSV_FILE['heart_above'],
    DATA_CSV_FILE['respiration_below'],
    ],
    columns = ['min_age',
    'max_age',
    'low_systolic',
    'low_diastolic',
    'high_systolic',
    'high_diastolic',
    'heart_below',
    'heart_above',
    'respiration_below',
    ])
Y = DATA_CSV_FILE['allow_ride']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

allow = clf.predict([[age, age, blood_pressure, blood_pressure, blood_pressure, blood_pressure, heart_rate, heart_rate, respiration]])

##For suggestion

Y = DATA_CSV_FILE['suggestion']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

suggestion = clf.predict([[age, age, blood_pressure, blood_pressure, blood_pressure, blood_pressure, heart_rate, heart_rate, respiration]])
suggestion = suggestion[0]
print(suggestion)