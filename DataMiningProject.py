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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns

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

fileLoc = "C:\\Users\\Noah\\Desktop\\DataMining\\"

df = pd.read_csv(fileLoc + 'NFL Play by Play 2009-2018 (v5).csv', usecols = selectedFeatures)

# filter 2018 games and only 4th downs
df2018 = df.where(df['game_id'] > 2018000000)
df2018 = df2018.where(df2018['down'] == 4)

df2018 = df2018.dropna(how='all')
#print(df2018.head()['down'])


def_down = pd.read_csv(fileLoc + 'defense_downs_2017.csv')
off_down = pd.read_csv(fileLoc + 'offense_downs_2017.csv')
spec_teams = pd.read_csv(fileLoc + 'special_teams_2017.csv')
off_rush = pd.read_csv(fileLoc + 'offense_rushing_2017.csv')
off_pass = pd.read_csv(fileLoc + 'offense_passing_2017.csv')
def_rush = pd.read_csv(fileLoc + 'defense_rush_2017.csv')
def_pass = pd.read_csv(fileLoc + 'defense_pass_2017.csv')


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
# sum = 0
# for i in range(10):
#     model = RandomForestClassifier(max_depth=12, max_features=25, min_samples_leaf=1, n_estimators=150)

#     model.fit(X_train, y_train)

#     #rfe = RFE(estimator = model, n_features_to_select = 15)
#     #fit = rfe.fit(X_train, y_train)

#     #results = model.predict(X_test)

#     results = model.predict(X_test)

#     # potential imbalance, calculate f1 score
#     f1 = f1_score(y_test, results)
#     sum += f1

# print(sum/10)

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

# Gradient Boosting Classifier with GridSearch
# Tried other parameters as well, but it took a long time to run
# so I trimmed it down in this file
# param_grid_gb = {
#     'n_estimators': [300, 500],
#     'max_depth': [3, 5],
#     'min_samples_leaf': [2, 3]
#     }
# gb_tuned = GradientBoostingClassifier()
# grid_search_gb = GridSearchCV(estimator=gb_tuned, param_grid=param_grid_gb, cv=3)
# grid_search_gb.fit(X_train, y_train)
# print(grid_search_gb.best_params_)

# # The best parameters given were max_depth=3, min_samples_leaf=3, n_estimators=300
# f1_sum = 0
# for i in range(10):
model = GradientBoostingClassifier(max_depth=3, min_samples_leaf=3, n_estimators=300)
model.fit(X_train, y_train)
gb_pred = model.predict(X_test)
f1 = f1_score(y_test, gb_pred)
#f1_sum += f1
# print(f1_sum / 10)

# # KNN without SMOTE, does not perform well
# knn_tuned = KNeighborsClassifier()
# param_grid_knn = {
#     'leaf_size': [3, 5, 10, 20],
#     'n_neighbors': [2, 3, 5],
#     'p': [1,2]}
# grid_search_knn = GridSearchCV(estimator=knn_tuned, param_grid=param_grid_knn)
# grid_search_knn.fit(X_train,y_train)
# print(grid_search_knn.best_params_)
# f1_sum = 0
# for i in range(10):
#     knn = KNeighborsClassifier(leaf_size=3, n_neighbors=2, p=1)
#     knn.fit(X_train, y_train)
#     knn_pred = knn.predict(X_test)
#     f1 = f1_score(y_test, knn_pred)
#     f1_sum += f1
# print(f1_sum / 10)

# # KNN with SMOTE no hyperparameter tuning
# SMOTE_sample = SMOTE()
# X_train_SMOTE, y_train_SMOTE = SMOTE_sample.fit_resample(X_train, y_train)
# f1_sum=0
# for i in range(10):
#     knn = KNeighborsClassifier()
#     knn.fit(X_train_SMOTE, y_train_SMOTE)
#     knn_pred = knn.predict(X_test)
#     f1 = f1_score(y_test, knn_pred)
#     f1_sum += f1
# print(f1_sum/10)

# # With hyperparameter tuning
# param_grid_knn = {
#     'n_neighbors': [2, 3, 5],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'minkowski', 'manhattan']}
# knn = KNeighborsClassifier()
# grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn)
# grid_search_knn.fit(X_train_SMOTE, y_train_SMOTE)
# print(grid_search_knn.best_params_)
# f1_sum=0
# for i in range(10):
#     knn = KNeighborsClassifier(metric='manhattan', n_neighbors=2, weights='distance')
#     knn.fit(X_train_SMOTE, y_train_SMOTE)
#     knn_pred = knn.predict(X_test)
#     f1 = f1_score(y_test, knn_pred)
#     f1_sum += f1
# print(f1_sum/10)

# # Decision Tree with hyperparameter tuning
# dtree = DecisionTreeClassifier()
# param_grid_dtree = {
#     'max_depth': [3, 5, 7, 10],
#     'min_samples_leaf': [1, 2, 3, 5],
#     'criterion': ["gini", "entropy"]
#     }
# grid_search_dt = GridSearchCV(estimator=dtree, param_grid=param_grid_dtree, cv=3, scoring='f1')
# grid_search_dt.fit(X_train, y_train)
# print(grid_search_dt.best_params_)

# f1_sum = 0
# for i in range(10):
#     dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, criterion='entropy')
#     dt.fit(X_train, y_train)
#     dt_pred = dt.predict(X_test)
#     f1 = f1_score(y_test, dt_pred)
#     f1_sum += f1
# print(f1_sum / 10)

# # SVM with SMOTE and hyperparameter tuning
# param_grid_svm = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'poly']}
# svm = SVC()
# grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm)
# grid_search_svm.fit(X_train_SMOTE, y_train_SMOTE)
# print(grid_search_svm.best_params_)
# f1_sum = 0
# for i in range(10):
#     svm = SVC(C=10, gamma=0.001)
#     svm.fit(X_train_SMOTE, y_train_SMOTE)
#     svm_pred = svm.predict(X_test)
#     f1_sum += f1_score(y_test, svm_pred)
# print(f1_sum/10)

# # Printing the 10 most correlated columns with Decision
# print(abs(df2018.corr()['decision']).sort_values(ascending = False).head(11))

# # Printing the 10 least correlated columns with Decision
# print(abs(df2018.corr()['decision']).sort_values().head(10))

# # Will print out every column's average value for rows that are 0 and 1
# # We have too many columns to get good output using this though
# print(df2018.groupby(['decision']).mean())

# # Can use this line to print specific columns so that we can view the columns we want to see
# print(df2018.groupby(['decision'])[['qtr', 'drive', 'posteam_score', 'defteam_score', 'no_score_prob', 'fg_prob']].mean())

# Bar chart for yds to go split up by decision
yds_decision = sns.barplot(data=df2018, x='decision', y='ydstogo')
yds_decision.set_title("Average Yards to go by Decision")
plt.show()

# Bar chart to plot the number of 0's and 1's for each quarter
quarter = sns.countplot(data=df2018, x='qtr', hue='decision')
quarter.set_title("Count of Decision by Quarter")
plt.show()

# Bar chart reflecting decision affect on win probability
yds_wp = sns.barplot(data=df2018, x='decision', y='wp', hue='decision')
yds_wp.set_title("Average Win Probability by Decision")
plt.show()

# Scatter plot showcasing the decision density around different scoring scenarios for the team with the ball vs the defending team
score_diff = sns.scatterplot(data=df2018, x='posteam_score', y='defteam_score', hue='decision')
score_diff.set_title("Score Differential and Decision")
plt.show()

# Print out the most important features coorelating 
feature_importances = pd.DataFrame({'Features': X_train.columns, 'feature_importance': model.feature_importances_})
print(feature_importances.sort_values('feature_importance', ascending=False).head(10))
feature_importances = feature_importances.sort_values('feature_importance', ascending=False).head(10)
plt.barh(feature_importances['Features'], feature_importances['feature_importance'])
plt.title('Feature Importances for Gradient Boosting')
plt.show()

# Seaborn barplot for feature importances
sns.barplot(data=feature_importances, x='feature_importance', y='Features')
plt.show()

# Top recorded average scores for the different models we tested (after running each 10 times)
model_scores = {
    'Decision Tree': 0.8494983277591974,
    'Random Forest': 0.8614390237532819,
    'Gradient Boosting': 0.8759873211,
    'KNN' : 0.37489435456,
    'SVM' : 0.284266543791
}

df = pd.DataFrame(model_scores.items())
model_perf = sns.barplot(data=df, x=0, y =1)
model_perf.set(xlabel = 'Models', ylabel='F score')
plt.show()



