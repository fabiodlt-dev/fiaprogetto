


def extract_features(df, team_stats, direct_comparisons):
    """Aggiunge classifica, media gol fatti/subiti, forma e confronti diretti al dataset."""
    df['HomeRank'] = df['HomeTeam'].map(lambda x: team_stats[x]['points'] if x in team_stats else 0)
    df['AwayRank'] = df['AwayTeam'].map(lambda x: team_stats[x]['points'] if x in team_stats else 0)

    # Media mobile dei gol fatti e subiti (ultime 5 partite)
    df['HomeGF'] = df['HomeTeam'].map(
        lambda x: sum(team_stats[x]['gf']) / 5 if x in team_stats and len(team_stats[x]['gf']) >= 5 else 0)
    df['AwayGF'] = df['AwayTeam'].map(
        lambda x: sum(team_stats[x]['gf']) / 5 if x in team_stats and len(team_stats[x]['gf']) >= 5 else 0)
    df['HomeGA'] = df['HomeTeam'].map(
        lambda x: sum(team_stats[x]['ga']) / 5 if x in team_stats and len(team_stats[x]['ga']) >= 5 else 0)
    df['AwayGA'] = df['AwayTeam'].map(
        lambda x: sum(team_stats[x]['ga']) / 5 if x in team_stats and len(team_stats[x]['ga']) >= 5 else 0)

    # Forma delle ultime 5 partite (W=3, D=1, L=0)
    df['HomeForm'] = df['HomeTeam'].map(lambda x: sum(
        [3 if r == 'W' else 1 if r == 'D' else 0 for r in team_stats[x]['form']]) if x in team_stats else 0)
    df['AwayForm'] = df['AwayTeam'].map(lambda x: sum(
        [3 if r == 'W' else 1 if r == 'D' else 0 for r in team_stats[x]['form']]) if x in team_stats else 0)

    # Confronti diretti
    df['HomeWins'] = df.apply(
        lambda row: direct_comparisons.get((row['HomeTeam'], row['AwayTeam']), {}).get('home_wins', 0), axis=1)
    df['AwayWins'] = df.apply(
        lambda row: direct_comparisons.get((row['HomeTeam'], row['AwayTeam']), {}).get('away_wins', 0), axis=1)
    df['Draws'] = df.apply(lambda row: direct_comparisons.get((row['HomeTeam'], row['AwayTeam']), {}).get('draws', 0),
                           axis=1)

    return df