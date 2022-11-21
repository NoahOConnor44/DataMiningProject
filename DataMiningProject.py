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
#print(df2018.head()['down'])


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

def decision(dataframe):
    list = []

    for row in dataframe['play_type']:
        if row == 'run' or row == 'pass':
            list.append(1)
        else:
            list.append(0)
    
    dataframe['decision'] = list
    
    return dataframe

df2018 = decision(df2018)

# remove strings from kick stats for each team on the year
df2018['Lng_y_y'] = df2018['Lng_y_y'].str.replace('T','').astype(float)
df2018['Lng_x_y'] = df2018['Lng_x_y'].str.replace('T','').astype(float)
df2018['Lng_y_x'] = df2018['Lng_y_x'].str.replace('T','').astype(float)
df2018['Lng'] = df2018['Lng'].str.replace('T','').astype(float)

# filter out data that has null/nan columns
df2018 = df[df['posteam'].notna()]

def changeToPercentage(dataframe):
    
    columns = ['20-29 > A-M', '30-39 > A-M','40-49 > A-M', '50+ > A-M']
    

    #print(dataframe.loc[:,columns])
    '''
    for column in columns:
        list = []

        for row in dataframe[column]:
            try:
                denom,num = str(row).split('_')
                #print(denom,num)
                total = float(num)/float(denom)

                list.append(total)
            except ValueError:
                print()    
            
        return dataframe
    '''



#df2018 = changeToPercentage(df2018)

print(df2018['posteam'].value_counts())
print(df2018['defteam'].value_counts())

#print(df2018.head()[['decision','play_type']])
#print(df2018.head())

# test = df2018.select_dtypes(include=['O']).keys

# print(test)





#X = df2018.drop(['decision'], axis=1)
#y = df2018['decision']



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)