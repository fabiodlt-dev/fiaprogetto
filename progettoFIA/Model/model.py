import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# Funzione di addestramento del modello
def train_model():
    # Caricare i dataset delle tre stagioni
    datasets = [
        pd.read_csv("datasetsMatch/season-2425.csv"),
        pd.read_csv("datasetsMatch/season-2324.csv"),
        pd.read_csv("datasetsMatch/season-2223.csv")
    ]

    # Unire tutti i dataset in uno solo
    dataset = pd.concat(datasets, ignore_index=True)

    # Rimuovere colonne non necessarie
    columns_to_drop = ['Date', 'Referee', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY',
                       'AY', 'HR', 'AR', 'FTHG', 'FTAG']
    dataset = dataset.drop(columns=columns_to_drop, errors='ignore')

    # Creiamo una lista di tutte le squadre presenti almeno una volta in 3 stagioni
    all_teams = pd.concat([dataset['HomeTeam'], dataset['AwayTeam']]).unique()

    # Codifica delle squadre con TUTTE le squadre esistenti
    le = LabelEncoder()
    le.fit(all_teams)  # Ora LabelEncoder conosce tutte le squadre

    dataset['HomeTeam'] = le.transform(dataset['HomeTeam'])
    dataset['AwayTeam'] = le.transform(dataset['AwayTeam'])

    # Creazione della feature target (esito della partita)
    dataset['Result'] = dataset['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    dataset = dataset.drop(columns=['FTR'])

    # Selezione delle feature e target
    X = dataset[['HomeTeam', 'AwayTeam']]
    y = dataset['Result']

    # Divisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Addestramento del modello con probabilità
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
    calibrated_model.fit(X_train, y_train)

    # Calcoliamo l'accuratezza
    y_pred = calibrated_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return calibrated_model, le, accuracy

# Funzione di previsione
def predict_outcome(model, le, home_team, away_team):
    try:
        home_encoded = le.transform([home_team])[0]
        away_encoded = le.transform([away_team])[0]
    except ValueError:
        return None

    # Creiamo un DataFrame con le stesse colonne di X_train
    new_match = pd.DataFrame([[home_encoded, away_encoded]], columns=['HomeTeam', 'AwayTeam'])

    # Previsione dell'esito della partita
    predicted_outcome = model.predict(new_match)[0]

    # Mappiamo i risultati come richiesto
    result_mapping = {0: "1", 1: "X", 2: "2"}
    return result_mapping[predicted_outcome]
