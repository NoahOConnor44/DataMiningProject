import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
df2018 = df2018[df2018['posteam'].notna()]

# creates new seperate columns for kicks attempted vs kicks made
df2018[['20-29 A','20-29 M']] = df2018['20-29 > A-M'].str.split('_', expand=True).astype('int')
df2018[['30-39 A','30-39 M']] = df2018['30-39 > A-M'].str.split('_', expand=True).astype('int')
df2018[['40-49 A','40-49 M']] = df2018['40-49 > A-M'].str.split('_', expand=True).astype('int')
df2018[['50+ A','50+ M']] = df2018['50+ > A-M'].str.split('_', expand=True).astype('int')

df2018 = df2018.drop(['1-19 > A-M', '20-29 > A-M', '30-39 > A-M', '40-49 > A-M', '50+ > A-M'], axis=1)
df2018 = df2018.drop(['posteam', 'defteam','posteam_type', 'play_type', 'Team_x', 'Team_y'], axis=1)

# remove rows with na values
df2018 = df2018.dropna()

# print(df2018.columns)
# for col in df2018.columns:
#     print(col)
# print(df2018['20-29 > A-M'])

#df2018 = changeToPercentage(df2018)

#print(df2018.head()[['decision','play_type']])
#print(df2018.head())

#test = df2018.select_dtypes(include=['O']).keys

X = df2018.drop(['decision'], axis=1)
y = df2018['decision']

# for col in df2018.columns:
#     print(df2018[col])

#X.to_numpy() to convert to numpy array
# select 50 best features from dataset
# print(X.shape)
#X_new = SelectKBest(f_classif, k=50).fit_transform(X.to_numpy(),y.to_numpy())
#np.seterr(divide='ignore', invalid='ignore')
# np.std(X, axis=0) == 0

# split data 70% into training, 30% into testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#model = LogisticRegression()
# model = RandomForestClassifier()
sum = 0
for i in range(10):
    model = RandomForestClassifier(max_depth=12, max_features=25, min_samples_leaf=1, n_estimators=150)
    model.fit(X_train, y_train)

    #rfe = RFE(estimator = model, n_features_to_select = 15)
    #fit = rfe.fit(X_train, y_train)

    #results = model.predict(X_test)

    results = model.predict(X_test)

    # potential imbalance, calculate f1 score
    f1 = f1_score(y_test, results)
    sum += f1

print(sum/10)

# param_grid = {
#     'max_depth': [12, 15, 20, 25],
#     'max_features': [12, 15, 25],
#     'min_samples_leaf': [1, 3, 5],
#     'n_estimators': [75, 100, 150]
# }

# rf = RandomForestClassifier()
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv=3)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)


#model.fit(X_train, y_train)
#results = model.predict(X_test)
#print(f1_score(y_test, results))