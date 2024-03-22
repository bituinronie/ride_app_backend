from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/', methods = ['GET'])
def send_message_from_root():
    return jsonify({"message":"If you see this message, it means the server is working correctly."})

@app.route('/train-model', methods = ['GET'])
def train_model():
    #Algorithm here - Decision Tree
    #import libraries
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    import joblib

    # Load the data into a pandas dataframe
    DATA_CSV_FILE = pd.read_csv('ride_data_set.csv')
    DATA_CSV_FILE.isnull().sum()

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

    # Train the model
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)

    # Save the trained model as a .pkl file
    joblib.dump(clf, "trained_model.pkl")

    return jsonify({"message: ":"Model trained and saved as trained_model.pkl"}) 

@app.route('/get-suggestion', methods = ['GET', 'POST'])
def recommend_suggest():
    if request.method == 'POST':
        try:
            #import libraries
            import joblib
            import numpy as np
            import pandas as pd

            # Get the data from the POST request.
            age = request.form.get('age')
            blood_pressure_systolic = request.form.get('blood_pressure_systolic')
            blood_pressure_diastolic = request.form.get('blood_pressure_diastolic')
            heart_rate = request.form.get('heart_rate')
            respiration = request.form.get('respiration')
            
            DATA_CSV_FILE = pd.read_csv('ride_data_set.csv')

            # Load the model from the saved .pkl file
            clf = joblib.load("trained_model.pkl")

            allow = clf.predict([[age, blood_pressure_systolic, blood_pressure_diastolic, heart_rate, respiration]])
            allow = allow[0]

            #Suggestions
            suggestions = DATA_CSV_FILE[DATA_CSV_FILE['allow_ride'] == allow]
            suggestions = suggestions.head(1)
            suggestion = suggestions['suggestion'].values[0]

            return jsonify({"allow":allow, "suggestions":suggestion})
        except Exception as e:
            #return what is the error from the try block
            return jsonify({"Error: ":str(e)})
    else:
        pass
        return jsonify({"Error: ":'Please submit the fields first.'})




if __name__ == "__main__":
    app.run(debug=True)


##Run Flask in custom host: flask run --host=127.1.1.1