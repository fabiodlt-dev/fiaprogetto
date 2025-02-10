from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

def train_model(dataset):
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

    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3,
                                       scoring='accuracy',
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

    return calibrated_model