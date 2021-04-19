"""
Created on Mon Apr 19 2021 22:10

@author: Sanjeev
"""

import pandas as pd

# read data from csv file into a pandas dataFrame
with open('.//Dataset//ipl_csv2//all_matches.csv') as f:
    ipl_data = pd.read_csv(f)

# print all columns
# print(ipl_data.columns)
# all columns
#     ['match_id', 'season', 'start_date', 'venue', 'innings', 'ball',
#        'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
#        'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
#        'penalty', 'wicket_type', 'player_dismissed', 'other_wicket_type',
#        'other_player_dismissed']

relevantColumns = ['match_id', 'venue', 'innings', 'ball',
    'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
    'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
    'penalty']

ipl_data = ipl_data[relevantColumns]

# print(ipl_data.head(3))

# create another column that tells the number of runs scored, including off the bat and 
# extra runs concended by the bowling team
ipl_data['totalRuns'] = ipl_data['runs_off_bat'] + ipl_data['extras']

# now drop the columns 'runs_off_bat' and 'extras' as they are not required anymore.
ipl_data = ipl_data.drop(columns=['runs_off_bat', 'extras'])

# only select rows belonging to first 6 overs
ipl_data = ipl_data[ipl_data['ball']<=5.6]

ipl_data = ipl_data[ipl_data['innings']<=2]

# preprocess the data so that we get a tuple of following kind in each row:
    # ('match_id', 'venue', 'innings', 'batting_team', 'bowling_team', 'totalRuns')
ipl_data = ipl_data.groupby(['match_id',
                                'venue',
                                'innings',
                                'batting_team',
                                'bowling_team']).totalRuns.sum()

# print(ipl_data.head(3))

# convert back to dataFrame
ipl_data = ipl_data.reset_index()
ipl_data = ipl_data.drop(columns=['match_id'])

# print(ipl_data.head(3))

ipl_data.to_csv("myPreprocessed.csv", index=False)