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
        from sklearn.tree import DecisionTreeClassifier

        age = request.form.get('age')
        blood_pressure_systolic = request.form.get('blood_pressure_systolic')
        blood_pressure_diastolic = request.form.get('blood_pressure_diastolic')
        heart_rate = request.form.get('heart_rate')
        respiration = request.form.get('respiration')
        
        DATA_CSV_FILE = pd.read_csv('ride_data_set.csv')
        DATA_CSV_FILE.isnull().sum()

        print(DATA_CSV_FILE)
        X = pd.DataFrame(np.c_[
            DATA_CSV_FILE['age'],
            DATA_CSV_FILE['systolic'],
            DATA_CSV_FILE['diastolic'],
            DATA_CSV_FILE['heart_rate'],
            DATA_CSV_FILE['respiration_rate'],
            ],
            columns = ['age',
            'systolic',
            'diastolic',
            'heart_rate',
            'respiration_rate',
            ])
        Y = DATA_CSV_FILE['allow_ride']

        clf = DecisionTreeClassifier()
        clf.fit(X, Y)

        allow = clf.predict([[age, blood_pressure_systolic, blood_pressure_diastolic, heart_rate, respiration]])
        allow = allow[0]

        #Suggestions
        suggestion = DATA_CSV_FILE[DATA_CSV_FILE['allow_ride'] == allow]
        suggestion = suggestion.head(1)
        suggestion = suggestion._get_value(0, 'suggestion')

        return jsonify({"allow":allow, "suggestions":suggestion})
    else:
        pass
        return jsonify({"Error: ":'Please submit the fields first.'})


if __name__ == "__main__":
    app.run(debug=True)


##Run Flask in custom host: flask run --host=127.1.1.1