import tkinter as tk
from tkinter import ttk, messagebox
from Model.model import train_model, predict_outcome

class BettingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Previsione Partite Calcio")
        self.root.geometry("500x400")
        self.root.configure(bg="#2E4053")

        # Caricamento del modello
        self.model, self.le, self.accuracy, self.team_stats, self.direct_comparisons = train_model()

        self.squadre = sorted(self.le.classes_)

        # Creazione interfaccia grafica
        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(self.root, text="Previsione Risultati Partite", font=("Arial", 16), bg="#2E4053", fg="white")
        title_label.pack(pady=10)

        self.frame = tk.Frame(self.root, bg="#2E4053")
        self.frame.pack(pady=5)

        self.home_var = tk.StringVar()
        self.away_var = tk.StringVar()

        self.home_menu = ttk.Combobox(self.frame, textvariable=self.home_var, values=self.squadre, width=20)
        self.home_menu.grid(row=0, column=0, padx=5, pady=5)

        self.away_menu = ttk.Combobox(self.frame, textvariable=self.away_var, values=self.squadre, width=20)
        self.away_menu.grid(row=0, column=1, padx=5, pady=5)

        self.add_button = tk.Button(self.frame, text="Prevedi", command=self.add_match, bg="#28B463", fg="white")
        self.add_button.grid(row=0, column=2, padx=5, pady=5)

        self.listbox_results = tk.Listbox(self.root, width=50, height=10)
        self.listbox_results.pack(pady=10)

        self.button_accuracy = tk.Button(self.root, text="Mostra Accuratezza", command=self.show_accuracy, bg="#D4AC0D", fg="black")
        self.button_accuracy.pack(pady=5)

        self.button_clear = tk.Button(self.root, text="Pulisci Partite", command=self.clear_matches, bg="#E74C3C", fg="white")
        self.button_clear.pack(pady=5)

    def add_match(self):
        home_team = self.home_var.get()
        away_team = self.away_var.get()

        if home_team and away_team and home_team != away_team:
            predicted_outcome = predict_outcome(self.model, self.le, home_team, away_team, self.team_stats, self.direct_comparisons)

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
