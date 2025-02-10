import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_features(dataset):
    dataset['HomeGoalsLast5'] = dataset.groupby('HomeTeam')['FTHG'].rolling(5, min_periods=1).sum().reset_index(level=0,
                                                                                                                drop=True)
    dataset['AwayGoalsLast5'] = dataset.groupby('AwayTeam')['FTAG'].rolling(5, min_periods=1).sum().reset_index(level=0,
                                                                                                                drop=True)
    dataset['HomeGoalsConcededLast5'] = dataset.groupby('HomeTeam')['FTAG'].rolling(5, min_periods=1).sum().reset_index(
        level=0, drop=True)
    dataset['AwayGoalsConcededLast5'] = dataset.groupby('AwayTeam')['FTHG'].rolling(5, min_periods=1).sum().reset_index(
        level=0, drop=True)
    dataset['HomeGoalDifferenceLast5'] = dataset['HomeGoalsLast5'] - dataset['HomeGoalsConcededLast5']
    dataset['AwayGoalDifferenceLast5'] = dataset['AwayGoalsLast5'] - dataset['AwayGoalsConcededLast5']

    standings = {}
    for season in [pd.read_csv("../datasetsMatch/season-2223.csv"), pd.read_csv("../datasetsMatch/season-2324.csv"),
                   pd.read_csv("../datasetsMatch/season-2122.csv"), pd.read_csv("../datasetsMatch/season-2021.csv")]:
        for _, row in season.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            if row['FTR'] == 'H':
                standings[home_team] = standings.get(home_team, 0) + 3
                standings[away_team] = standings.get(away_team, 0)
            elif row['FTR'] == 'A':
                standings[home_team] = standings.get(home_team, 0)
                standings[away_team] = standings.get(away_team, 0) + 3
            else:
                standings[home_team] = standings.get(home_team, 0) + 1
                standings[away_team] = standings.get(away_team, 0) + 1

    dataset['HomeTeamPoints'] = dataset['HomeTeam'].map(standings)
    dataset['AwayTeamPoints'] = dataset['AwayTeam'].map(standings)

    dataset['HomeWinsAgainstAwayTeam'] = 0
    dataset['AwayWinsAgainstHomeTeam'] = 0
    dataset['DrawsAgainstEachOther'] = 0

    for i, row in dataset.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        head_to_head = dataset[((dataset['HomeTeam'] == home_team) & (dataset['AwayTeam'] == away_team)) |
                               ((dataset['HomeTeam'] == away_team) & (dataset['AwayTeam'] == home_team))]

        home_wins = head_to_head[head_to_head['Result'] == 0].shape[0]
        away_wins = head_to_head[head_to_head['Result'] == 2].shape[0]
        draws = head_to_head[head_to_head['Result'] == 1].shape[0]

        dataset.at[i, 'HomeWinsAgainstAwayTeam'] = home_wins
        dataset.at[i, 'AwayWinsAgainstHomeTeam'] = away_wins
        dataset.at[i, 'DrawsAgainstEachOther'] = draws

    home_performance = dataset.groupby('HomeTeam').agg(
        home_goals_scored=('FTHG', 'mean'),
        home_goals_conceded=('FTAG', 'mean'),
        home_games=('HomeTeam', 'size')
    )
    away_performance = dataset.groupby('AwayTeam').agg(
        away_goals_scored=('FTAG', 'mean'),
        away_goals_conceded=('FTHG', 'mean'),
        away_games=('AwayTeam', 'size')
    )

    dataset = dataset.merge(home_performance, on='HomeTeam', how='left')
    dataset = dataset.merge(away_performance, on='AwayTeam', how='left')

    # Normalizzazione Min-Max
    scaler = MinMaxScaler()
    features_to_normalize = ['HomeGoalsLast5', 'AwayGoalsLast5', 'HomeGoalsConcededLast5', 'AwayGoalsConcededLast5',
                             'HomeGoalDifferenceLast5', 'AwayGoalDifferenceLast5', 'HomeTeamPoints', 'AwayTeamPoints',
                             'HomeWinsAgainstAwayTeam', 'AwayWinsAgainstHomeTeam', 'DrawsAgainstEachOther',
                             'home_goals_scored', 'home_goals_conceded', 'home_games',
                             'away_goals_scored', 'away_goals_conceded', 'away_games']

    dataset[features_to_normalize] = dataset[features_to_normalize].fillna(0)
    dataset[features_to_normalize] = scaler.fit_transform(dataset[features_to_normalize])

    return dataset
