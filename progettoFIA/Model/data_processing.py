import pandas as pd

def load_season_data():
    """Carica i dati delle stagioni passate e della stagione attuale."""
    past_seasons = [
        pd.read_csv("datasetsMatch/season-2021.csv"),
        pd.read_csv("datasetsMatch/season-2122.csv"),
        pd.read_csv("datasetsMatch/season-2223.csv"),
        pd.read_csv("datasetsMatch/season-2324.csv")

    ]
    current_season = pd.read_csv("datasetsMatch/season-2425.csv")

    # Combina le stagioni passate in un unico DataFrame
    past_data = pd.concat(past_seasons, ignore_index=True)

    return past_data, current_season

def compute_team_stats(df):
    """Calcola classifica, gol fatti/subiti, forma recente e confronti diretti delle squadre."""
    df = df.sort_values(by=['Date'])
    team_stats = {}
    direct_comparisons = {}  # Dizionario per i confronti diretti

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
        match_key = tuple(sorted([home, away]))  # Usa una chiave ordinata per evitare duplicati
        if match_key not in direct_comparisons:
            direct_comparisons[match_key] = {'home_wins': 0, 'away_wins': 0, 'draws': 0}

        if result == 'H':
            direct_comparisons[match_key]['home_wins'] += 1
        elif result == 'A':
            direct_comparisons[match_key]['away_wins'] += 1
        else:
            direct_comparisons[match_key]['draws'] += 1

    return team_stats, direct_comparisons
