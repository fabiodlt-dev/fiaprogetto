import tkinter as tk
from tkinter import ttk, messagebox
from Model.model import addestra_modello as train_model, predici_risultato as predict_outcome

class BettingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Previsione Partite Calcio")
        self.root.geometry("500x400")
        self.root.configure(bg="#2E4053")

        # Train the model and unpack the return values
        try:
            (self.model, self.le, self.accuracy, self.team_ranking,
             self.home_goals, self.away_goals, self.total_goals, self.scaler) = train_model()
        except ValueError as e:
            messagebox.showerror("Errore", f"Errore durante il caricamento del modello: {e}")
            self.root.quit()  # Exit the app if unpacking fails

        # Sorting teams alphabetically for the dropdown
        self.squadre = sorted(self.le.classes_)
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = tk.Label(
            self.root,
            text="Previsione Risultati Partite",
            font=("Arial", 16),
            bg="#2E4053",
            fg="white"
        )
        title_label.pack(pady=10)

        # Frame for comboboxes and buttons
        self.frame = tk.Frame(self.root, bg="#2E4053")
        self.frame.pack(pady=5)

        # Variables for home and away teams
        self.home_var = tk.StringVar()
        self.away_var = tk.StringVar()

        # Home team combobox
        self.home_menu = ttk.Combobox(
            self.frame,
            textvariable=self.home_var,
            values=self.squadre,
            width=20
        )
        self.home_menu.grid(row=0, column=0, padx=5, pady=5)

        # Away team combobox
        self.away_menu = ttk.Combobox(
            self.frame,
            textvariable=self.away_var,
            values=self.squadre,
            width=20
        )
        self.away_menu.grid(row=0, column=1, padx=5, pady=5)

        # Predict button
        self.add_button = tk.Button(
            self.frame,
            text="Prevedi",
            command=self.add_match,
            bg="#28B463",
            fg="white"
        )
        self.add_button.grid(row=0, column=2, padx=5, pady=5)

        # Listbox to display the results
        self.listbox_results = tk.Listbox(self.root, width=50, height=10)
        self.listbox_results.pack(pady=10)

        # Button to show accuracy
        self.button_accuracy = tk.Button(
            self.root,
            text="Mostra Accuratezza",
            command=self.show_accuracy,
            bg="#D4AC0D",
            fg="black"
        )
        self.button_accuracy.pack(pady=5)

        # Button to clear predictions
        self.button_clear = tk.Button(
            self.root,
            text="Pulisci Partite",
            command=self.clear_matches,
            bg="#E74C3C",
            fg="white"
        )
        self.button_clear.pack(pady=5)

    def add_match(self):
        home_team = self.home_var.get()
        away_team = self.away_var.get()

        # Ensure that both home and away teams are selected and are not the same
        if home_team and away_team and home_team != away_team:
            # Retrieve relevant data for predictions
            home_ranking = self.team_ranking.get(home_team, None)
            away_ranking = self.team_ranking.get(away_team, None)
            home_gs = self.home_goals.get(home_team, 0)
            away_gs = self.away_goals.get(away_team, 0)

            if home_ranking is None or away_ranking is None:
                messagebox.showerror("Errore", "Una o entrambe le squadre non hanno un ranking valido.")
                return

            # Predict match outcome
            predicted_outcome = predict_outcome(
                self.model,
                self.le,
                self.scaler,
                home_team,
                away_team,
                home_ranking,
                away_ranking,
                home_gs,
                away_gs
            )

            if predicted_outcome:
                result_text = f'{home_team} vs {away_team}: {predicted_outcome}'
                self.listbox_results.insert(tk.END, result_text)
            else:
                messagebox.showerror("Errore", "Impossibile ottenere la previsione.")
        else:
            messagebox.showwarning("Attenzione", "Seleziona due squadre diverse.")

    def show_accuracy(self):
        messagebox.showinfo("Accuratezza del Modello", f"L'accuratezza del modello è: {self.accuracy * 100:.2f}%")

    def clear_matches(self):
        self.listbox_results.delete(0, tk.END)

def create_gui():
    root = tk.Tk()
    app = BettingApp(root)
    return root

if __name__ == "__main__":
    root = create_gui()
    root.mainloop()
