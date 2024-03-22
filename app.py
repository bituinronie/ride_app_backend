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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import joblib

    # Load the data into a pandas dataframe
    DATA_CSV_FILE = pd.read_csv('data_is_fit.csv')
    DATA_CSV_FILE = DATA_CSV_FILE.dropna()

    le = LabelEncoder()
    DATA_CSV_FILE['Time_of_Day'] = le.fit_transform(DATA_CSV_FILE['Time_of_Day'])

    X = DATA_CSV_FILE.drop('Is_Fit', axis=1)
    Y = DATA_CSV_FILE['Is_Fit']

    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, Y)

    # Save the trained model as a .pkl file
    joblib.dump(clf, "trained_model.pkl")

    return jsonify({"message: ":"Model trained and saved as trained_model.pkl"}) 

@app.route('/get-suggestion', methods = ['GET', 'POST'])
def recommend_suggest():
    if request.method == 'POST':
        # Heart_Rate_(BPM)
        # HRV_(ms)
        # Systolic_BP_(mmHg)
        # Diastolic_BP_(mmHg)
        # Respiration_Rate_(Breaths_per_Minute)
        # Blood_Oxygen_Level_(SpO2)
        # Ambient_Temperature_(C)
        # Ambient_Noise_Level_(dB)
        # Time_of_Day
        # Previous_Activity_Level_(Steps)
        # Age
        # Is_Fit
        try:
            #import libraries
            import joblib
            import numpy as np
            import pandas as pd

            # Get the data from the POST request.
            heart_rate_bpm = request.form.get('heart_rate_bpm')
            hrv = request.form.get('hrv')
            systolic_bp = request.form.get('systolic_bp')
            diastolic_bp = request.form.get('diastolic_bp')
            respiration_rate = request.form.get('respiration_rate')
            blood_oxygen_level = request.form.get('blood_oxygen_level')
            ambient_temperature = request.form.get('ambient_temperature')
            ambient_noise_level = request.form.get('ambient_noise_level')
            time_of_day = request.form.get('time_of_day')
            previous_activity_level = request.form.get('previous_activity_level')
            age = request.form.get('age')
            

            # Get the data from the POST request.
            # age = request.form.get('age')
            # blood_pressure_systolic = request.form.get('blood_pressure_systolic')
            # blood_pressure_diastolic = request.form.get('blood_pressure_diastolic')
            # heart_rate = request.form.get('heart_rate')
            # respiration = request.form.get('respiration')
            
            DATA_CSV_FILE = pd.read_csv('ride_data_set.csv')

            # Load the model from the saved .pkl file
            clf = joblib.load("trained_model.pkl")

            allow = clf.predict([[heart_rate_bpm, hrv, systolic_bp, diastolic_bp, respiration_rate, blood_oxygen_level, ambient_temperature, ambient_noise_level, time_of_day, previous_activity_level, age]])
            allow = allow[0]

            if allow == 1:
                allow = 'YES'
            else:
                allow = 'NO'

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