
def read_data():
    df = pd.read_csv('DataMining\\NFL Play by Play 2009-2018 (v5).csv', usecols = selectedFeatures)
    def_down = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\defense_downs_2017.csv')
    off_down = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\offense_downs_2017.csv')
    spec_teams = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\special_teams_2017.csv')
    off_rush = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\offense_rushing_2017.csv')
    off_pass = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\offense_passing_2017.csv')
    def_rush = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\defense_rush_2017.csv')
    def_pass = pd.read_csv('C:\\Users\\Noah\\Desktop\\DataMining\\defense_pass_2017.csv')