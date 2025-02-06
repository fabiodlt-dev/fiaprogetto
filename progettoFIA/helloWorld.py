import pandas as pd

# Leggi il dataset della stagione attuale di Serie A
dataset = pd.read_csv("season-2425.csv")

# Imposta opzioni di visualizzazione
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
asdfasdfsdf
# Creazione del DataFrame
df = pd.DataFrame(dataset)

# Data cleaning
newDf = df.drop(['HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'HTHG', 'HTAG'], axis='columns')

newDf = newDf.rename(columns={'Date': 'Data', 'HomeTeam': 'TeamCasa', 'AwayTeam': 'TeamTrasferta', 'FTHG': 'TeamCasaGol',
                              'FTAG': 'TeamTrasfertaGol', 'FTR': 'EsitoPartita'})


print(newDf)


# Calcolo del ranking attuale basato sulle vittorie (esempio)
# Prima conta il numero di vittorie per ogni squadra






