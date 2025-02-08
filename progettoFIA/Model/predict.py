import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from data_processing import load_season_data, compute_team_stats
from feature_engineering import extract_features

def train_model():
    """Carica i dati, addestra il modello e calcola l'accuratezza."""
    past_seasons = [

        load_season_data("datasetsMatch/season-2021.csv"),
        load_season_data("datasetsMatch/season-2122.csv"),
        load_season_data("datasetsMatch/season-2223.csv"),
        load_season_data("datasetsMatch/season-2324.csv")
    ]
    current_season = load_season_data("datasetsMatch/season-2425.csv")

    past_data = pd.concat(past_seasons, ignore_index=True)
    team_stats, direct_comparisons = compute_team_stats(past_data)

    current_season = extract_features(current_season, team_stats, direct_comparisons)

    le = LabelEncoder()
    all_teams = pd.concat([current_season['HomeTeam'], current_season['AwayTeam']]).unique()
    le.fit(all_teams)
    current_season['HomeTeam'] = le.transform(current_season['HomeTeam'])
    current_season['AwayTeam'] = le.transform(current_season['AwayTeam'])

    current_season['Result'] = current_season['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    current_season = current_season.drop(columns=['FTR', 'Date'])

    X = current_season[['HomeTeam', 'AwayTeam', 'HomeRank', 'AwayRank', 'HomeGF', 'AwayGF', 'HomeGA', 'AwayGA', 'HomeForm', 'AwayForm', 'HomeWins', 'AwayWins', 'Draws']]
    y = current_season['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
    calibrated_model.fit(X_train, y_train)

    y_pred = calibrated_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return calibrated_model, le, accuracy, team_stats, direct_comparisons