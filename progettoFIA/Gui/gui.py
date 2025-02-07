import tkinter as tk
from tkinter import messagebox
from Model.model import train_model, predict_outcome

# Addestramento del modello
model, le, accuracy = train_model()

# Funzione per aggiungere le partite e ottenere le previsioni
def add_match():
    home_team = entry_home_team.get()
    away_team = entry_away_team.get()

    if home_team and away_team:
        predicted_outcome = predict_outcome(model, le, home_team, away_team)
        if predicted_outcome:
            result_text = f'{home_team} vs {away_team}: {predicted_outcome}'
            listbox_results.insert(tk.END, result_text)
        else:
            messagebox.showerror("Errore", "Impossibile ottenere la previsione.")
    else:
        messagebox.showwarning("Campo vuoto", "Inserisci entrambe le squadre.")

# Funzione per ottenere l'accuratezza
def show_accuracy():
    messagebox.showinfo("Accuratezza del Modello", f"L'accuratezza del modello è: {accuracy * 100:.2f}%")

# Funzione per rimuovere tutte le partite dalla lista
def clear_matches():
    listbox_results.delete(0, tk.END)

# Creazione della finestra principale
root = tk.Tk()
root.title("Previsione Partite Calcio")

# Etichette e campi di inserimento
label_home_team = tk.Label(root, text="Squadra di Casa:")
label_home_team.pack(pady=5)

entry_home_team = tk.Entry(root)
entry_home_team.pack(pady=5)

label_away_team = tk.Label(root, text="Squadra Ospite:")
label_away_team.pack(pady=5)

entry_away_team = tk.Entry(root)
entry_away_team.pack(pady=5)

# Bottone per aggiungere la partita
button_add_match = tk.Button(root, text="Aggiungi Partita", command=add_match)
button_add_match.pack(pady=5)

# Lista per visualizzare i risultati delle partite
listbox_results = tk.Listbox(root, width=50, height=10)
listbox_results.pack(pady=10)

# Bottone per mostrare l'accuratezza
button_accuracy = tk.Button(root, text="Mostra Accuratezza", command=show_accuracy)
button_accuracy.pack(pady=5)

# Bottone per cancellare tutte le partite
button_clear = tk.Button(root, text="Pulisci Partite", command=clear_matches)
button_clear.pack(pady=5)

# Avvio della finestra
root.mainloop()
