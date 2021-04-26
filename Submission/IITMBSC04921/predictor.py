"""
Created on Tue Apr 20 2021 01:31

@author: Sanjeev
"""

### Custom definitions and classes if any ###
import pandas as pd
import numpy as np
import joblib

def predictRuns(testInput):
    
    with open('regressionModel.joblib', 'rb') as f:
        regressor = joblib.load(f)
    with open('venueEncoder.joblib', 'rb') as f:
        venueEncoder = joblib.load(f)
    with open('teamEncoder.joblib', 'rb') as f:
        teamEncoder = joblib.load(f)

    # rest test data
    testCase = pd.read_csv(testInput)

    # rename team
    testCase['batting_team'] = testCase['batting_team'].replace('Punjab Kings', 'Kings XI Punjab')
    testCase['bowling_team'] = testCase['bowling_team'].replace('Punjab Kings', 'Kings XI Punjab')
    
    # # Delhi Daredevils
    # testCase['batting_team'] = testCase['batting_team'].replace('Delhi Capitals', 'Delhi Daredevils')
    # testCase['bowling_team'] = testCase['bowling_team'].replace('Delhi Capitals', 'Delhi Daredevils')

    # encode venue and batting & bowling teams
    testCase['venue']       = venueEncoder.transform(testCase['venue'])
    testCase['batting_team']= teamEncoder.transform(testCase['batting_team'])
    testCase['bowling_team']= teamEncoder.transform(testCase['bowling_team'])

    # make sure that the order of columns is same as that fed to model
    testCase = testCase[['venue', 'innings', 'batting_team', 'bowling_team']]

    # convert input test case into numpy array
    testArray = testCase.to_numpy()

    # one hot encode venue, batting and bowling teams
    testCase = np.concatenate((np.eye(42)[testArray[:,0]],
                                np.eye(2)[testArray[:,1] -1 ],
                                np.eye(15)[testArray[:,2]],
                                np.eye(15)[testArray[:,3]],
                                ),
                                axis = 1)

    prediction = regressor.predict(testCase)
    ### Your Code Here ###
    prediction = 7*int(prediction[0])
    prediction = prediction//2 if prediction>70 else prediction*2 if prediction<20 else prediction
    return prediction
