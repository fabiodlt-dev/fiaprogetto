import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint


def save_final_dataset(dataset, filename="final_dataset.csv"):
    """
    Salva il dataset finale in un file CSV.
    """
    dataset.to_csv(filename, index=False)
    print(f"Dataset finale salvato in {filename}")


def save_match_data(match_data, filename="match_data.csv"):
    """
    Salva i dati della partita in un file CSV.
    """
    match_data.to_csv(filename, index=False)
    print(f"Dati della partita salvati in {filename}")


def train_model():
    # Caricare i dataset delle tre stagioni
    datasets = [
        pd.read_csv("datasetsMatch/season-2425.csv"),
        pd.read_csv("datasetsMatch/season-2324.csv"),
        pd.read_csv("datasetsMatch/season-2223.csv"),
        pd.read_csv("datasetsMatch/season-2122.csv"),
        pd.read_csv("datasetsMatch/season-2021.csv")
    ]

    # Unire tutti i dataset in uno solo
    dataset = pd.concat(datasets, ignore_index=True)

    # Rimuovere colonne non necessarie
    columns_to_drop = ['Date', 'Referee', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY',
                       'AY', 'HR', 'AR']
    dataset = dataset.drop(columns=columns_to_drop, errors='ignore')

    # Aggiungiamo nuove feature
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

    # Codifica delle squadre con TUTTE le squadre esistenti
    all_teams = pd.concat([dataset['HomeTeam'], dataset['AwayTeam']]).unique()
    le = LabelEncoder()
    le.fit(all_teams)  # Ora LabelEncoder conosce tutte le squadre

    dataset['HomeTeam'] = le.transform(dataset['HomeTeam'])
    dataset['AwayTeam'] = le.transform(dataset['AwayTeam'])

    # Creazione della feature target (esito della partita)
    dataset['Result'] = dataset['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    dataset = dataset.drop(columns=['FTR'])

    # Calcolare la classifica (punti per squadra)
    standings = {}
    for season in [pd.read_csv("datasetsMatch/season-2223.csv"), pd.read_csv("datasetsMatch/season-2324.csv"), pd.read_csv(
            "datasetsMatch/season-2122.csv"), pd.read_csv("datasetsMatch/season-2021.csv")]:
        for _, row in season.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            if row['FTR'] == 'H':  # Home win
                standings[home_team] = standings.get(home_team, 0) + 3
                standings[away_team] = standings.get(away_team, 0)
            elif row['FTR'] == 'A':  # Away win
                standings[home_team] = standings.get(home_team, 0)
                standings[away_team] = standings.get(away_team, 0) + 3
            else:  # Draw
                standings[home_team] = standings.get(home_team, 0) + 1
                standings[away_team] = standings.get(away_team, 0) + 1

    dataset['HomeTeamPoints'] = dataset['HomeTeam'].map(standings)
    dataset['AwayTeamPoints'] = dataset['AwayTeam'].map(standings)

    # Aggiungere le statistiche testa a testa
    dataset['HomeWinsAgainstAwayTeam'] = 0
    dataset['AwayWinsAgainstHomeTeam'] = 0
    dataset['DrawsAgainstEachOther'] = 0

    for i, row in dataset.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        # Calcolare i risultati testa a testa
        head_to_head = dataset[((dataset['HomeTeam'] == home_team) & (dataset['AwayTeam'] == away_team)) |
                               ((dataset['HomeTeam'] == away_team) & (dataset['AwayTeam'] == home_team))]

        home_wins = head_to_head[head_to_head['Result'] == 0].shape[0]  # Vittorie in casa
        away_wins = head_to_head[head_to_head['Result'] == 2].shape[0]  # Vittorie in trasferta
        draws = head_to_head[head_to_head['Result'] == 1].shape[0]  # Pareggi

        dataset.at[i, 'HomeWinsAgainstAwayTeam'] = home_wins
        dataset.at[i, 'AwayWinsAgainstHomeTeam'] = away_wins
        dataset.at[i, 'DrawsAgainstEachOther'] = draws

    # Rendimento in casa e in trasferta
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

    # Salva il dataset finale in un file CSV
    save_final_dataset(dataset)

    # Selezione delle feature e target
    X = dataset[
        ['HomeTeam', 'AwayTeam', 'HomeGoalsLast5', 'AwayGoalsLast5', 'HomeGoalsConcededLast5', 'AwayGoalsConcededLast5',
         'HomeGoalDifferenceLast5', 'AwayGoalDifferenceLast5', 'HomeTeamPoints', 'AwayTeamPoints',
         'home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded',
         'HomeWinsAgainstAwayTeam', 'AwayWinsAgainstHomeTeam', 'DrawsAgainstEachOther']]
    y = dataset['Result']

    # Divisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Addestramento del modello con RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(100, 300),  # Numero di alberi
        'max_depth': [None, 10, 20, 30],  # Profondità massima degli alberi
        'min_samples_split': randint(2, 10)  # Numero minimo di campioni per dividere un nodo
    }
    model = RandomForestClassifier(random_state=42, class_weight='balanced')

    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy',
                                       random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Miglior modello trovato
    best_model = random_search.best_estimator_

    # Calibrazione del modello
    calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid')
    calibrated_model.fit(X_train, y_train)

    # Calcolare l'accuratezza sul test set
    y_pred = calibrated_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🔹 Accuratezza del modello: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    return calibrated_model, le, dataset


def create_match_data(le, home_team, away_team, dataset):
    """
    Crea un DataFrame con tutti i dati della partita.
    """
    try:
        home_encoded = le.transform([home_team])[0]
        away_encoded = le.transform([away_team])[0]
    except ValueError:
        print(" Errore: Una delle squadre inserite non è presente nei dati.")
        return None

    # Creiamo un DataFrame con le stesse colonne di X_train
    match_data = pd.DataFrame([[home_encoded, away_encoded, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              columns=['HomeTeam', 'AwayTeam', 'HomeGoalsLast5', 'AwayGoalsLast5',
                                       'HomeGoalsConcededLast5',
                                       'AwayGoalsConcededLast5', 'HomeGoalDifferenceLast5', 'AwayGoalDifferenceLast5',
                                       'HomeTeamPoints', 'AwayTeamPoints', 'home_goals_scored', 'home_goals_conceded',
                                       'away_goals_scored', 'away_goals_conceded', 'HomeWinsAgainstAwayTeam',
                                       'AwayWinsAgainstHomeTeam', 'DrawsAgainstEachOther'])

    # Aggiungiamo i dati calcolati dal dataset originale
    home_stats = dataset[dataset['HomeTeam'] == home_encoded].iloc[-1]  # Prendi l'ultima partita della squadra di casa
    away_stats = dataset[dataset['AwayTeam'] == away_encoded].iloc[-1]  # Prendi l'ultima partita della squadra ospite

    match_data['HomeGoalsLast5'] = home_stats['HomeGoalsLast5']
    match_data['AwayGoalsLast5'] = away_stats['AwayGoalsLast5']
    match_data['HomeGoalsConcededLast5'] = home_stats['HomeGoalsConcededLast5']
    match_data['AwayGoalsConcededLast5'] = away_stats['AwayGoalsConcededLast5']
    match_data['HomeGoalDifferenceLast5'] = home_stats['HomeGoalDifferenceLast5']
    match_data['AwayGoalDifferenceLast5'] = away_stats['AwayGoalDifferenceLast5']
    match_data['HomeTeamPoints'] = home_stats['HomeTeamPoints']
    match_data['AwayTeamPoints'] = away_stats['AwayTeamPoints']
    match_data['home_goals_scored'] = home_stats['home_goals_scored']
    match_data['home_goals_conceded'] = home_stats['home_goals_conceded']
    match_data['away_goals_scored'] = away_stats['away_goals_scored']
    match_data['away_goals_conceded'] = away_stats['away_goals_conceded']
    match_data['HomeWinsAgainstAwayTeam'] = home_stats['HomeWinsAgainstAwayTeam']
    match_data['AwayWinsAgainstHomeTeam'] = away_stats['AwayWinsAgainstHomeTeam']
    match_data['DrawsAgainstEachOther'] = home_stats['DrawsAgainstEachOther']

    return match_data


def predict_probabilities(model, le, home_team, away_team):
    """
    Funzione per prevedere le probabilità di una partita.
    """
    try:
        home_encoded = le.transform([home_team])[0]
        away_encoded = le.transform([away_team])[0]
    except ValueError:
        print(" Errore: Una delle squadre inserite non è presente nei dati.")
        return None

    # Creiamo un DataFrame con le stesse colonne di X_train
    new_match = pd.DataFrame([[home_encoded, away_encoded, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                             columns=['HomeTeam', 'AwayTeam', 'HomeGoalsLast5', 'AwayGoalsLast5',
                                      'HomeGoalsConcededLast5',
                                      'AwayGoalsConcededLast5', 'HomeGoalDifferenceLast5', 'AwayGoalDifferenceLast5',
                                      'HomeTeamPoints', 'AwayTeamPoints', 'home_goals_scored', 'home_goals_conceded',
                                      'away_goals_scored', 'away_goals_conceded', 'HomeWinsAgainstAwayTeam',
                                      'AwayWinsAgainstHomeTeam', 'DrawsAgainstEachOther'])

    probabilities = model.predict_proba(new_match)[0]
    result_mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    return {result_mapping[i]: round(probabilities[i] * 100, 2) for i in range(3)}


# Addestriamo il modello con i 3 dataset
model, le, dataset = train_model()

# Input dell'utente
home_team = "Napoli"
away_team = "Inter"

# Creiamo i dati della partita
match_data = create_match_data(le, home_team, away_team, dataset)

# Salviamo i dati della partita in un file CSV
if match_data is not None:
    save_match_data(match_data, "match_data.csv")

# Previsione delle probabilità
probabilities = predict_probabilities(model, le, home_team, away_team)

if probabilities:
    print(f' Probabilità per {home_team} vs {away_team}:')
    for outcome, probability in probabilities.items():
        print(f' {outcome}: {probability}%')