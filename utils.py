from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def decision(dataframe):
    list = []

    for row in dataframe['play_type']:
        if row == 'run' or row == 'pass':
            list.append(1)
        else:
            list.append(0)
    
    dataframe['decision'] = list
    
    return dataframe


def teams(): 
    return {
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

# Creates a training and testing set
def create_X_y(df):
    X = df.drop(['decision'], axis=1)
    y = df['decision']
    return X,y

# Takes in a model, evalutes its performance over 10 runs. Returns the avg f1 score
def get_average_f1(model,df,Smote=True):
    X,y = create_X_y(df)
    f1_sum = 0
    for i in range(10):
        print('Creating training and test data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        if Smote:
            SMOTE_sample = SMOTE()
            X_train,y_train = SMOTE_sample.fit_resample(X_train, y_train)
       
        print('Fitting X train and y train')
        model.fit(X_train, y_train)
        print('Getting Results')
        results = model.predict(X_test)
        # potential imbalance, calculate f1 score
        f1 = f1_score(y_test, results)
        print('F1 score is ', f1, ' for trial ', i+1)
        f1_sum += f1
        print()
    print('Average F1 score is ', f1_sum/10)
    print('-----------------------------------------------\n')
    return(f1_sum/10)

def avg_yards_decision_graph(df):
    # Bar chart for yds to go split up by decision
    yds_decision = sns.barplot(data=df, x='decision', y='ydstogo')
    yds_decision.set_title("Average Yards to go by Decision")
    plt.show()


def qtr_decision_graph(df):
    # Bar chart to plot the number of 0's and 1's for each quarter
    quarter = sns.countplot(data=df, x='qtr', hue='decision')
    quarter.set_title("Count of Decision by Quarter")
    plt.show()


def win_probability_graph(df):
    # Bar chart reflecting decision affect on win probability
    yds_wp = sns.barplot(data=df, x='decision', y='wp', hue='decision')
    yds_wp.set_title("Average Win Probability by Decision")
    plt.show()
  


def scoring_decision_graph(df):
    # Scatter plot showcasing the decision density around different scoring scenarios for the team with the ball vs the defending team
    score_diff = sns.scatterplot(data=df, x='posteam_score', y='defteam_score', hue='decision')
    score_diff.set_title("Score Differential and Decision")
    plt.show()


def avg_scores_graph(scores):
    df = pd.DataFrame(scores.items())
    model_perf = sns.barplot(data=df, x=0, y =1)
    model_perf.set(xlabel = 'Models', ylabel='F score')
    plt.show()
    
def feature_importance_graph(df):
    X,y = create_X_y(df)
    # Print out the most important features coorelating 
    model = GradientBoostingClassifier(max_depth=3, min_samples_leaf=3, n_estimators=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train,y_train)
    feature_importances = pd.DataFrame({'Features': X_train.columns, 'feature_importance': model.feature_importances_})
    # print(feature_importances.sort_values('feature_importance', ascending=False).head(10))
    feature_importances = feature_importances.sort_values('feature_importance', ascending=False).head(10)
    barh = plt.barh(feature_importances['Features'], feature_importances['feature_importance'])
    plt.title('Feature Importances for Gradient Boosting')
    plt.show()

    # Seaborn barplot for feature importances
    sns_plot = sns.barplot(data=feature_importances, x='feature_importance', y='Features')
    plt.show()

    


def plot_all(df,scores):
    
    avg_yards_decision_graph(df)
    qtr_decision_graph(df)
    win_probability_graph(df)
    scoring_decision_graph(df)
    avg_scores_graph(scores)
    print('Next 2 graphs are feature importance and take a minute to show...')
    feature_importance_graph(df)
   