from flask import Flask, jsonify, request
from flask_cors import CORS

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/', methods = ['GET'])
def get_articles():
    return jsonify({"Hello":"World"})

#Set the recommended program here
recommendedProgram = "NA"

@app.route('/get-suggestion', methods = ['GET', 'POST'])
def recommend_program():
    if request.method == 'POST':
        #Algorithm here - Decision Tree
        #import libraries
        import numpy as np
        import pandas as pd  
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        age = request.form.get('age')
        blood_pressure_systolic = request.form.get('blood_pressure_systolic')
        blood_pressure_diastolic = request.form.get('blood_pressure_diastolic')
        heart_rate = request.form.get('heart_rate')
        respiration = request.form.get('respiration')
        print(request)
        
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

        allow = clf.predict([[age, age, blood_pressure_systolic, blood_pressure_diastolic, blood_pressure_systolic, blood_pressure_diastolic, heart_rate, heart_rate, respiration]])
        allow = allow[0]

        #Suggestions
        Y = DATA_CSV_FILE['suggestion']


        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, Y_train)

        suggestion = clf.predict([[age, age, blood_pressure_systolic, blood_pressure_diastolic, blood_pressure_systolic, blood_pressure_diastolic, heart_rate, heart_rate, respiration]])
        suggestion = suggestion[0]

        return jsonify({"allow":allow, "suggestions":suggestion})
    else:
        pass
        return jsonify({"Error: ":'Please submit the fields first.'})


if __name__ == "__main__":
    app.run(debug=True)


##Run Flask in custom host: flask run --host=127.1.1.1