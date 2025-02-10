import pandas as pd

def predict_probabilities(model, le, home_team, away_team):
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

# Esempio di utilizzo
if __name__ == "__main__":
    from data_processing import load_and_preprocess_data
    from feature_engineering import add_features
    from model import train_model

    # Caricare e preprocessare i dati
    dataset, le = load_and_preprocess_data()

    # Aggiungere feature
    dataset = add_features(dataset)

    # Addestrare il modello
    model = train_model(dataset)

    # Input dell'utente
    home_team = input("Inserisci la squadra di casa: ")
    away_team = input("Inserisci la squadra ospite: ")

    # Previsione delle probabilità
    probabilities = predict_probabilities(model, le, home_team, away_team)

    if probabilities:
        print(f' Probabilità per {home_team} vs {away_team}:')
        for outcome, probability in probabilities.items():
            print(f' {outcome}: {probability}%')
