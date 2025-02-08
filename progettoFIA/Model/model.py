import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from progettoFIA.Model.feature_engineering import extract_features


def compute_team_stats(df):
    """Calcola classifica, gol fatti/subiti, forma recente e confronti diretti delle squadre."""
    df = df.sort_values(by=['Date'])
    team_stats = {}
    direct_comparisons = {}  # Dizionario per memorizzare i confronti diretti

    for _, row in df.iterrows():
        home, away, result = row['HomeTeam'], row['AwayTeam'], row['FTR']
        home_goals, away_goals = row['FTHG'], row['FTAG']

        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {'points': 0, 'gf': [], 'ga': [], 'form': '', 'matches_played': 0}

        # Aggiorna punti e forma
        if result == 'H':
            team_stats[home]['points'] += 3
            team_stats[home]['form'] = 'W' + team_stats[home]['form'][:4]
            team_stats[away]['form'] = 'L' + team_stats[away]['form'][:4]
        elif result == 'A':
            team_stats[away]['points'] += 3
            team_stats[away]['form'] = 'W' + team_stats[away]['form'][:4]
            team_stats[home]['form'] = 'L' + team_stats[home]['form'][:4]
        else:
            team_stats[home]['points'] += 1
            team_stats[away]['points'] += 1
            team_stats[home]['form'] = 'D' + team_stats[home]['form'][:4]
            team_stats[away]['form'] = 'D' + team_stats[away]['form'][:4]

        # Aggiorna gol
        team_stats[home]['gf'].append(home_goals)
        team_stats[home]['ga'].append(away_goals)
        team_stats[away]['gf'].append(away_goals)
        team_stats[away]['ga'].append(home_goals)

        # Mantieni solo le ultime 5 partite
        team_stats[home]['gf'] = team_stats[home]['gf'][-5:]
        team_stats[home]['ga'] = team_stats[home]['ga'][-5:]
        team_stats[away]['gf'] = team_stats[away]['gf'][-5:]
        team_stats[away]['ga'] = team_stats[away]['ga'][-5:]

        # Aggiorna i confronti diretti
        if (home, away) not in direct_comparisons:
            direct_comparisons[(home, away)] = {'home_wins': 0, 'away_wins': 0, 'draws': 0}
        if result == 'H':
            direct_comparisons[(home, away)]['home_wins'] += 1
        elif result == 'A':
            direct_comparisons[(home, away)]['away_wins'] += 1
        else:
            direct_comparisons[(home, away)]['draws'] += 1

    return team_stats, direct_comparisons

def train_model():
    past_seasons = [
        pd.read_csv("datasetsMatch/season-2324.csv"),
        pd.read_csv("datasetsMatch/season-2223.csv")
    ]
    current_season = pd.read_csv("datasetsMatch/season-2425.csv")

    past_data = pd.concat(past_seasons, ignore_index=True)
    team_stats, direct_comparisons = compute_team_stats(past_data)  # Usa solo le stagioni passate per le statistiche

    current_season = extract_features(current_season, team_stats, direct_comparisons)  # Applica le statistiche alla stagione corrente

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

    # Ottimizzazione degli iperparametri con GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid')
    calibrated_model.fit(X_train, y_train)

    y_pred = calibrated_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return calibrated_model, le, accuracy, team_stats, direct_comparisons

def predict_outcome(model, le, home_team, away_team, team_stats, direct_comparisons):
    try:
        home_encoded = le.transform([home_team])[0]
        away_encoded = le.transform([away_team])[0]
    except ValueError:
        return None

    # Crea un nuovo DataFrame con le feature necessarie
    new_match = pd.DataFrame([[home_encoded, away_encoded,
                               team_stats.get(home_team, {}).get('points', 0),
                               team_stats.get(away_team, {}).get('points', 0),
                               sum(team_stats.get(home_team, {}).get('gf', [0])) / 5,
                               sum(team_stats.get(away_team, {}).get('gf', [0])) / 5,
                               sum(team_stats.get(home_team, {}).get('ga', [0])) / 5,
                               sum(team_stats.get(away_team, {}).get('ga', [0])) / 5,
                               sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in team_stats.get(home_team, {}).get('form', '')]),
                               sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in team_stats.get(away_team, {}).get('form', '')]),
                               direct_comparisons.get((home_team, away_team), {}).get('home_wins', 0),
                               direct_comparisons.get((home_team, away_team), {}).get('away_wins', 0),
                               direct_comparisons.get((home_team, away_team), {}).get('draws', 0)]],
                             columns=['HomeTeam', 'AwayTeam', 'HomeRank', 'AwayRank', 'HomeGF', 'AwayGF', 'HomeGA', 'AwayGA', 'HomeForm', 'AwayForm', 'HomeWins', 'AwayWins', 'Draws'])

    # Effettua la previsione
    predicted_outcome = model.predict(new_match)[0]

    # Mappa il risultato numerico in una stringa (1, X, 2)
    result_mapping = {0: "1", 1: "X", 2: "2"}
    return result_mapping[predicted_outcome]