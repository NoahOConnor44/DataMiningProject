import pandas as pd

selectedFeatures = ['game_id', 'posteam','qtr','posteam_type','defteam', 'down', 'quarter_seconds_remaining','yardline_100','drive','ydstogo','ydsnet','play_type','posteam_timeouts_remaining','defteam_timeouts_remaining','posteam_score','defteam_score','no_score_prob','fg_prob','td_prob', 'epa', 'wp']

fileLoc = "C:\\Users\\Noah\\Desktop\\DataMining\\"

df = pd.read_csv(fileLoc + 'NFL Play by Play 2009-2018 (v5).csv', usecols = selectedFeatures)