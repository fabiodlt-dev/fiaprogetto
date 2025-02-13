from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def add_features(dataset):
        # Assicuriamoci che il dataset sia ordinato (si assume che l'indice rifletta l'ordine cronologico delle partite)
        dataset = dataset.sort_index().reset_index(drop=True)
        # Creiamo un identificativo per ogni partita
        dataset['MatchID'] = dataset.index

        # --- Trasformazione in formato "long" per le apparizioni delle squadre ---
        # Per le partite in casa:
        home_df = dataset[['MatchID', 'HomeTeam', 'FTHG', 'FTAG']].copy()
        home_df.rename(columns={'HomeTeam': 'Team',
                                'FTHG': 'GoalsScored',
                                'FTAG': 'GoalsConceded'}, inplace=True)
        home_df['Venue'] = 'home'

        # Per le partite in trasferta:
        away_df = dataset[['MatchID', 'AwayTeam', 'FTAG', 'FTHG']].copy()
        away_df.rename(columns={'AwayTeam': 'Team',
                                'FTAG': 'GoalsScored',
                                'FTHG': 'GoalsConceded'}, inplace=True)
        away_df['Venue'] = 'away'

        # Uniamo i due dataframe in uno solo
        df_long = pd.concat([home_df, away_df]).sort_values('MatchID')

        # --- Calcolo dei gol segnati negli ultimi 5 match per ogni squadra ---
        df_long['RollingGoalsScored'] = df_long.groupby('Team')['GoalsScored'] \
            .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())

        # --- Ricostruzione delle feature per il dataset originale ---
        home_rolling = df_long[df_long['Venue'] == 'home'][['MatchID', 'RollingGoalsScored']] \
            .rename(columns={'RollingGoalsScored': 'HomeGoalsLast5'})
        away_rolling = df_long[df_long['Venue'] == 'away'][['MatchID', 'RollingGoalsScored']] \
            .rename(columns={'RollingGoalsScored': 'AwayGoalsLast5'})

        dataset = dataset.merge(home_rolling, on='MatchID', how='left')
        dataset = dataset.merge(away_rolling, on='MatchID', how='left')

        # --- Calcolare i gol concessi nelle ultime 5 partite per squadra ---
        df_long['RollingGoalsConceded'] = df_long.groupby('Team')['GoalsConceded'] \
            .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())

        home_conceded = df_long[df_long['Venue'] == 'home'][['MatchID', 'RollingGoalsConceded']] \
            .rename(columns={'RollingGoalsConceded': 'HomeGoalsConcededLast5'})
        away_conceded = df_long[df_long['Venue'] == 'away'][['MatchID', 'RollingGoalsConceded']] \
            .rename(columns={'RollingGoalsConceded': 'AwayGoalsConcededLast5'})

        dataset = dataset.merge(home_conceded, on='MatchID', how='left')
        dataset = dataset.merge(away_conceded, on='MatchID', how='left')

        # Calcolare la differenza gol nelle ultime 5 partite per squadra
        dataset['HomeGoalDifferenceLast5'] = dataset['HomeGoalsLast5'] - dataset['HomeGoalsConcededLast5']
        dataset['AwayGoalDifferenceLast5'] = dataset['AwayGoalsLast5'] - dataset['AwayGoalsConcededLast5']

        # Calcolare i punti in classifica per ogni squadra (solo per le prime 228 righe, stagione 2024-2025)
        standings = {team: 0 for team in pd.concat([dataset['HomeTeam'], dataset['AwayTeam']]).unique()}

        # Consideriamo solo la stagione 2024-2025 (prime 228 righe)
        season_2425 = pd.read_csv("../datasetsMatch/season-2425.csv").iloc[:228]

        for _, row in season_2425.iterrows():
            home_team, away_team = row['HomeTeam'], row['AwayTeam']

            # Aggiungiamo le squadre al dizionario standings se non esistono ancora
            if home_team not in standings:
                standings[home_team] = 0
            if away_team not in standings:
                standings[away_team] = 0

            # Aggiornare i punti
            if row['FTR'] == 'H':
                standings[home_team] += 3  # Vittoria per la squadra di casa
            elif row['FTR'] == 'A':
                standings[away_team] += 3  # Vittoria per la squadra in trasferta
            else:
                standings[home_team] += 1  # Pareggio per la squadra di casa
                standings[away_team] += 1  # Pareggio per la squadra in trasferta

        dataset['HomeTeamPoints'] = dataset['HomeTeam'].map(standings)
        dataset['AwayTeamPoints'] = dataset['AwayTeam'].map(standings)

        # Calcolare gli scontri diretti
        dataset['HomeWinsAgainstAwayTeam'] = 0
        dataset['AwayWinsAgainstHomeTeam'] = 0
        dataset['DrawsAgainstEachOther'] = 0

        # Modifica del calcolo per vittorie, sconfitte e pareggi nei confronti diretti
        for i, row in dataset.iterrows():
            home_team, away_team = row['HomeTeam'], row['AwayTeam']
            head_to_head = dataset[((dataset['HomeTeam'] == home_team) & (dataset['AwayTeam'] == away_team)) |
                                   ((dataset['HomeTeam'] == away_team) & (dataset['AwayTeam'] == home_team))]

            # Calcolare vittorie in casa, vittorie in trasferta e pareggi
            home_wins = head_to_head[head_to_head['Result'] == 0].shape[0]
            away_wins = head_to_head[head_to_head['Result'] == 2].shape[0]
            draws = head_to_head[head_to_head['Result'] == 1].shape[0]

            # Aggiornare il dataset con i valori calcolati
            dataset.at[i, 'HomeWinsAgainstAwayTeam'] = home_wins
            dataset.at[i, 'AwayWinsAgainstHomeTeam'] = away_wins
            dataset.at[i, 'DrawsAgainstEachOther'] = draws

        # Calcolare la media gol segnati e subiti per squadra
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

        # --- Normalizzazione delle variabili numeriche ---
        scaler = MinMaxScaler()

        # Selezionare solo le colonne numeriche per la normalizzazione
        columns_to_normalize = [
            'HomeGoalsLast5', 'AwayGoalsLast5', 'HomeGoalsConcededLast5', 'AwayGoalsConcededLast5',
            'HomeGoalDifferenceLast5', 'AwayGoalDifferenceLast5', 'HomeTeamPoints', 'AwayTeamPoints',
            'home_goals_scored', 'home_goals_conceded', 'away_goals_scored', 'away_goals_conceded',
            'HomeWinsAgainstAwayTeam', 'AwayWinsAgainstHomeTeam', 'DrawsAgainstEachOther'
        ]

        dataset[columns_to_normalize] = scaler.fit_transform(dataset[columns_to_normalize])

        # Riempire eventuali valori NaN con 0
        dataset.fillna(0, inplace=True)

        # Mantenere tutte le colonne originali
        original_columns = dataset.columns.tolist()



        return dataset
