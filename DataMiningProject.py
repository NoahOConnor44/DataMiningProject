import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

selectedFeatures = ['game_id', 'posteam','qtr','posteam_type','defteam', 'down', 'quarter_seconds_remaining','yardline_100','drive','ydstogo','ydsnet','play_type','posteam_timeouts_remaining','defteam_timeouts_remaining','posteam_score','defteam_score','no_score_prob','fg_prob','td_prob', 'epa', 'wp']

teams = {
    'Dolphins': 'MIA',
    '49ers' : 'SF',
    'Bears' : 'CHI',
    'Bengals' : 'CIN',
    'Bills' : 'BUF',
    'Broncos': 'DEN',
    'Browns' : 'CLE',
    'Buccaneers' : 'TB',
    'Cardinals' : 'ARI',
    'Chargers' : 'LAC',
    'Chiefs' : 'KC',
    'Colts' : 'IND',
    'Cowboys' : 'DAL',
    'Eagles' : 'PHI',
    'Falcons' : 'ATL',
    'Giants' : 'NYG',
    'Jaguars' : 'JAX',
    'Jets' : 'NYJ',
    'Lions' : 'DET',
    'Packers' : 'GB',
    'Panthers' : 'CAR',
    'Patriots' : 'NE',
    'Raiders' : 'OAK',
    'Rams' : 'LA',
    'Ravens' : 'BAL',
    'Redskins' : 'WAS',
    'Saints' : 'NO',
    'Seahawks' : 'SEA',
    'Steelers': 'PIT',
    'Texans': 'HOU',
    'Titans': 'TEN',
    'Vikings': 'MIN'
}

df = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\NFL Play by Play 2009-2018 (v5).csv', usecols = selectedFeatures)

# filter 2018 games and only 4th downs
df2018 = df.where(df['game_id'] > 2018000000)
df2018 = df2018.where(df2018['down'] == 4)

df2018 = df2018.dropna(how='all')
print(df2018.head()['down'])


def_down = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\defense_downs_2017.csv')
off_down = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\offense_downs_2017.csv')
spec_teams = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\special_teams_2017.csv')
off_rush = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\offense_rushing_2017.csv')
off_pass = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\offense_passing_2017.csv')
def_rush = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\defense_rush_2017.csv')
def_pass = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\defense_pass_2017.csv')


offense = off_down.merge(spec_teams, on='Team')
offense = offense.merge(off_rush, on='Team')
offense = offense.merge(off_pass, on='Team')
defense = def_down.merge(def_pass, on='Team')
defense = defense.merge(def_rush, on='Team')

offense = offense.replace({"Team": teams})
defense = defense.replace({"Team" : teams})

df2018 = df2018.merge(offense, how='left', left_on='posteam', right_on='Team')
df2018 = df2018.merge(defense, how='left', left_on='defteam', right_on='Team')

def decision(row):
    if row['play_type'] == 'run' or row['play_type'] == 'pass':
        return 1
    else:
        return 0

df2018 = df2018.apply(lambda row: decision(row), axis=1)

X = df2018.drop(['decision'], axis=1)
y = df2018['decision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(offense.head())
print(defense.head())