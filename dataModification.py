import pandas as pd
from utils import teams, decision

pd.options.mode.chained_assignment = None  # default='warn'


# Reads in all the data from projects datasets
def read_data(features):
    df = pd.read_csv('./data/NFL Play by Play 2009-2018 (v5).csv', usecols = features)
    def_down = pd.read_csv('./data/defense_downs_2017.csv')
    off_down = pd.read_csv('./data/offense_downs_2017.csv')
    spec_teams = pd.read_csv('./data/special_teams_2017.csv')
    off_rush = pd.read_csv('./data/offense_rushing_2017.csv')
    off_pass = pd.read_csv('./data/offense_passing_2017.csv')
    def_rush = pd.read_csv('./data/defense_rush_2017.csv')
    def_pass = pd.read_csv('./data/defense_pass_2017.csv')
    
    return df,def_down,off_down,spec_teams,off_rush,off_pass,def_rush,def_pass

# Get only data for 2018 and fourth downs
def filter_df(df):
    df2018 = df.where(df['game_id'] > 2018000000)
    df2018 = df2018.where(df2018['down'] == 4)
    return df2018.dropna(how='all')

# Gets offensive data into a single df 
def make_offense_df(off_down,spec_teams,off_rush, off_pass):
    offense = off_down.merge(spec_teams, on='Team')
    offense = offense.merge(off_rush, on='Team')
    offense = offense.merge(off_pass, on='Team')
    return offense.replace({"Team": teams()})

# Gets defensive data into a single df
def make_defensive_df(def_down,def_pass,def_rush):
    defense = def_down.merge(def_pass, on='Team')
    defense = defense.merge(def_rush, on='Team')
    return defense.replace({"Team" : teams()})

# Combines main df, and offense, and defensive df
def merge_all_df(df2018,off_down,spec_teams,off_rush, off_pass,def_down,def_pass,def_rush):
    offense = make_offense_df(off_down,spec_teams,off_rush, off_pass)
    defense = make_defensive_df(def_down,def_pass,def_rush)
    df2018 = df2018.merge(offense, how='left', left_on='posteam', right_on='Team')
    return df2018.merge(defense, how='left', left_on='defteam', right_on='Team')

# Handles issues with improper strings and adjusts kicking columns from format of attempts_made to their own columns of attempts and made
# Adds decision column of whether team went for it or not and dropped unneccessary columns
def adjust_data_issues(df2018):
    kicking_columns = ['Lng_y_y','Lng_x_y','Lng_y_x','Lng']
    new_kicking_columns = [['20-29 A','20-29 M'],['30-39 A','30-39 M'],['40-49 A','40-49 M'],['50+ A','50+ M']]
    old_kicking_columns = ['20-29 > A-M','30-39 > A-M','40-49 > A-M','50+ > A-M']
    drop_columns = ['posteam', 'defteam','posteam_type', 'play_type', 'Team_x', 'Team_y'] + old_kicking_columns
    
    df2018 = decision(df2018)
    
    # remove strings from kick stats for each team on the year
    for col in kicking_columns:
        df2018[col] = df2018[col].str.replace('T','').astype(float)
    
    # filter out data that has null/nan columns
    df2018 = df2018[df2018['posteam'].notna()]
    
    # creates new seperate columns for kicks attempted vs kicks made
    for i in range(len(new_kicking_columns)):
        df2018[new_kicking_columns[i]] = df2018[old_kicking_columns[i]].str.split('_', expand=True).astype('int')
        
    df2018 = df2018.drop(drop_columns, axis=1)
    # remove rows with na values
    return df2018.dropna()
 
    
def adjust_main_df():
    selectedFeatures = ['game_id', 'posteam','qtr','posteam_type','defteam',
                        'down', 'quarter_seconds_remaining','yardline_100',
                        'drive','ydstogo','ydsnet','play_type','posteam_timeouts_remaining',
                        'defteam_timeouts_remaining','posteam_score','defteam_score','no_score_prob',
                        'fg_prob','td_prob', 'epa', 'wp']
    
    df,def_down,off_down,spec_teams,off_rush,off_pass,def_rush,def_pass = read_data(selectedFeatures)
    df2018 = filter_df(df)
    df2018 = merge_all_df(df2018,off_down,spec_teams,off_rush, off_pass,def_down,def_pass,def_rush)
    return adjust_data_issues(df2018)

# df = adjust_main_df()
# print(df)

