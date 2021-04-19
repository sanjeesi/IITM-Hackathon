"""
Created on Tue Apr 20 2021 01:06

@author: Sanjeev
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# read data from unzipped file
data = pd.read_csv('myPreprocessed.csv')

venueEncoder = LabelEncoder()
teamEncoder = LabelEncoder()

data['venue'] = venueEncoder.fit_transform(data['venue'])
data['batting_team'] = teamEncoder.fit_transform(data['batting_team'])
data['bowling_team'] = teamEncoder.fit_transform(data['bowling_team'])

# get data in numpy array
anArray = data.to_numpy()

# get independent and target variables
X, y = anArray[:,:3], anArray[:,3]

X = np.concatenate((np.eye(42)[anArray[:,0]],
                        np.eye(2)[anArray[:,1] -1 ],
                        np.eye(15)[anArray[:,2]],
                        np.eye(15)[anArray[:,3]],
                        ), axis = 1)

# split data in training and testing.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.25)

# fit a linear regressor
linearRegressor = LinearRegression()

# train the model
linearRegressor.fit(X_train, y_train)

# save the model and supporting label encoders
joblib.dump(linearRegressor, 'regressionModel.joblib')
joblib.dump(venueEncoder, 'venueEncoder.joblib')
joblib.dump(teamEncoder, 'teamEncoder.joblib')

print(linearRegressor.score(X_test, y_test))