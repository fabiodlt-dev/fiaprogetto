import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    # Caricare i dataset delle stagioni
    datasets = [
        pd.read_csv("../datasetsMatch/season-2425.csv"),
        pd.read_csv("../datasetsMatch/season-2324.csv"),
        pd.read_csv("../datasetsMatch/season-2223.csv"),
        pd.read_csv("../datasetsMatch/season-2122.csv"),
        pd.read_csv("../datasetsMatch/season-2021.csv")
    ]

    # Unire tutti i dataset in uno solo
    dataset = pd.concat(datasets, ignore_index=True)

    # Rimuovere colonne non necessarie
    columns_to_drop = ['Date', 'Referee', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY',
                       'AY', 'HR', 'AR']
    dataset = dataset.drop(columns=columns_to_drop, errors='ignore')

    # Codifica delle squadre
    all_teams = pd.concat([dataset['HomeTeam'], dataset['AwayTeam']]).unique()
    le = LabelEncoder()
    le.fit(all_teams)  # LabelEncoder conosce tutte le squadre

    dataset['HomeTeam'] = le.transform(dataset['HomeTeam'])
    dataset['AwayTeam'] = le.transform(dataset['AwayTeam'])

    # Creazione della feature target (esito della partita)
    dataset['Result'] = dataset['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    dataset = dataset.drop(columns=['FTR'])

    return dataset, le