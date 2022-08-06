from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.utils.validation import check_array

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    Age = request.form['Age']
    AverageLeadTime = request.form['AverageLeadTime']
    DaysSinceCreation = request.form['DaysSinceCreation']
    PersonsNights = request.form['PersonsNights']
    RoomNights = request.form['RoomNights']
    DaysSinceLastStay = request.form['DaysSinceLastStay']
    daysSinceFirstStay = request.form['daysSinceFirstStay']
    DistributionChannel = request.form['DistributionChannel']
    MarketSegment = request.form['MarketSegment']
    arr = np.array([[Age, AverageLeadTime, DaysSinceCreation, PersonsNights, RoomNights, DaysSinceLastStay, daysSinceFirstStay, DistributionChannel, MarketSegment]])
    arr1 = check_array(arr)
    pred = model.predict(arr1)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)