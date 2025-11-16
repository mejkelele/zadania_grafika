import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import ttk, messagebox
import math


class BezierCurveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Krzywa Béziera - Kotkowo 🐱")
        self.root.geometry("1200x800")

        self.control_points = []
        self.dragging_point = None
        self.degree = 3
        self.resolution = 100
        self.entry_vars = []

        self.setup_gui()

    def setup_gui(self):
        """Setup interfejsu GUI"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lewy panel - kontrola
        control_frame = ttk.LabelFrame(main_frame, text="Kontrola Krzywej", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Prawy panel - rysowanie
        draw_frame = ttk.LabelFrame(main_frame, text="Płótno")
        draw_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Kontrola - stopień krzywej
        degree_frame = ttk.LabelFrame(control_frame, text="Stopień Krzywej (n)")
        degree_frame.pack(fill=tk.X, padx=5, pady=5)

        self.degree_var = tk.IntVar(value=3)
        ttk.Spinbox(
            degree_frame,
            from_=1,
            to=10,
            textvariable=self.degree_var,
            command=self.update_curve,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Kontrola - przyciski
        button_frame = ttk.LabelFrame(control_frame, text="Akcje")
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(button_frame, text="Wyczyść", command=self.clear_canvas).pack(
            fill=tk.X, padx=5, pady=2
        )
        ttk.Button(button_frame, text="Losuj Punkty", command=self.random_points).pack(
            fill=tk.X, padx=5, pady=2
        )
        ttk.Button(button_frame, text="Dodaj Punkt", command=self.add_point).pack(
            fill=tk.X, padx=5, pady=2
        )

        # Punkty kontrolne
        points_container = ttk.Frame(control_frame)
        points_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar dla punktów
        points_canvas = tk.Canvas(points_container, height=200)
        scrollbar = ttk.Scrollbar(
            points_container, orient="vertical", command=points_canvas.yview
        )
        self.points_frame = ttk.Frame(points_canvas)

        points_canvas.create_window((0, 0), window=self.points_frame, anchor="nw")
        points_canvas.configure(yscrollcommand=scrollbar.set)

        points_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.points_frame.bind(
            "<Configure>",
            lambda e: points_canvas.configure(scrollregion=points_canvas.bbox("all")),
        )

        # Rysowanie
        self.setup_canvas(draw_frame)

        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Kliknij na płótnie aby dodać punkty kontrolne")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.update_points_display()

    def setup_canvas(self, parent):
        """Setup płótna do rysowania"""
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_title("Krzywa Béziera - Przeciągaj punkty! 🐾")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Podpięcie eventów
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def binomial_coefficient(self, n, k):
        """Współczynnik dwumianowy $\binom{n}{k}$"""
        if k < 0 or k > n:
            return 0
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    def bernstein_polynomial(self, n, i, t):
        """Wielomian Bernsteina $B_{i,n}(t)$"""
        return self.binomial_coefficient(n, i) * (t**i) * ((1 - t) ** (n - i))

    def bezier_curve(self, points, num_points=100):
        """Oblicza punkty krzywej Béziera, wykorzystując de Casteljau lub Bernsteina"""
        n = len(points) - 1
        if n < 1:
            return np.array([])

        # Ograniczanie stopnia, jeśli punktów jest za mało
        if n < self.degree:
            pass

        # Aby krzywa pasowała do stopnia ze Spinboxa (jeśli jest więcej punktów niż stopień + 1)
        # Przy standardowej metodzie de Casteljau/Bernsteina, stopień jest zawsze = n.
        # Jeśli chcemy użyć tylko self.degree, musielibyśmy obcinać listę points, ale to wypacza ideę punktów kontrolnych.
        # Zostawiamy standardową definicję: stopień = liczba_punktów - 1.

        curve_points = []
        t_values = np.linspace(0, 1, num_points)

        for t in t_values:
            x, y = 0, 0
            for i, point in enumerate(points):
                # Wielomian Bernsteina (stopień n = len(points) - 1)
                bern = self.bernstein_polynomial(n, i, t)
                x += point[0] * bern
                y += point[1] * bern
            curve_points.append((x, y))

        return np.array(curve_points)

    def draw_curve(self):
        """Rysuje krzywą Béziera i punkty kontrolne"""
        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_title("Krzywa Béziera - Przeciągaj punkty! 🐾")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")

        # Rysuj krzywą jeśli są przynajmniej 2 punkty
        if len(self.control_points) >= 2:
            curve_points = self.bezier_curve(self.control_points, self.resolution)
            if len(curve_points) > 0:
                self.ax.plot(
                    curve_points[:, 0],
                    curve_points[:, 1],
                    "b-",
                    linewidth=2,
                    label=f"Krzywa stopnia {len(self.control_points) - 1}",
                )

        # Rysuj punkty kontrolne i linie pomocnicze
        if len(self.control_points) > 0:
            points_array = np.array(self.control_points)
            self.ax.plot(
                points_array[:, 0],
                points_array[:, 1],
                "ro--",
                alpha=0.5,
                linewidth=1,
                label="Linia kontrolna",
            )

            for i, (x, y) in enumerate(self.control_points):
                # Używamy kółek do zaznaczenia punktów
                circle = Circle((x, y), 0.1, color="red", alpha=0.7)
                self.ax.add_patch(circle)
                self.ax.text(x + 0.15, y + 0.15, f"P{i}", fontsize=8, color="darkred")

        # Odświeżanie stopnia krzywej w tytule
        if len(self.control_points) > 0:
            self.ax.legend(loc="upper left")

        self.canvas.draw()

    def on_click(self, event):
        """Obsługa kliknięcia myszą"""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        # Sprawdź czy kliknięto w istniejący punkt
        for i, point in enumerate(self.control_points):
            px, py = point
            # Promień dotyku (dopasowany do rozmiaru kółka)
            if abs(px - x) < 0.2 and abs(py - y) < 0.2:
                self.dragging_point = i
                self.status_var.set(f"Przeciągasz punkt P{i}")
                return

        # Dodaj nowy punkt
        self.control_points.append((x, y))
        self.status_var.set(f"Dodano punkt P{len(self.control_points)-1}")
        self.update_curve()
        self.update_points_display()

    def on_motion(self, event):
        """Obsługa ruchu myszą"""
        if event.inaxes != self.ax or self.dragging_point is None:
            return

        x, y = event.xdata, event.ydata
        self.control_points[self.dragging_point] = (x, y)
        self.update_curve()
        self.update_points_display()

    def on_release(self, event):
        """Obsługa zwolnienia przycisku myszy"""
        self.dragging_point = None
        self.status_var.set("Gotowy - przeciągaj punkty lub dodawaj nowe")

    def update_curve(self):
        """Aktualizacja krzywej (wywoływana przez Spinbox)"""
        # Stopień jest używany tylko w random_points i do wyświetlania w GUI
        self.degree = self.degree_var.get()
        self.draw_curve()
        # Ponowna aktualizacja wyświetlania punktów jest konieczna, jeśli wpisano z klawiatury w Spinbox,
        # co wymusiło update_curve, ale nie chcemy jej tu (jest w on_motion/on_click)

    def update_points_display(self):
        """Aktualizacja wyświetlania punktów kontrolnych (Tabela)"""
        # Wyczyść frame
        for widget in self.points_frame.winfo_children():
            widget.destroy()

        self.entry_vars = []

        # Dodaj nagłówek
        header_frame = ttk.Frame(self.points_frame)
        header_frame.pack(fill=tk.X, pady=2)
        ttk.Label(header_frame, text="Punkt", font="Arial 8 bold").pack(
            side=tk.LEFT, padx=5
        )
        ttk.Label(header_frame, text="X", font="Arial 8 bold").pack(
            side=tk.LEFT, padx=15
        )
        ttk.Label(header_frame, text="Y", font="Arial 8 bold").pack(
            side=tk.LEFT, padx=15
        )
        ttk.Label(header_frame, text="Akcje", font="Arial 8 bold").pack(
            side=tk.LEFT, padx=10
        )

        # Dodaj pola dla każdego punktu
        for i, (x, y) in enumerate(self.control_points):
            point_frame = ttk.Frame(self.points_frame, relief="solid", borderwidth=1)
            point_frame.pack(fill=tk.X, pady=1, padx=2)

            # Etykieta punktu
            ttk.Label(point_frame, text=f"P{i}", width=4).pack(side=tk.LEFT, padx=5)

            # Zmienne dla pól entry
            x_var = tk.DoubleVar(value=round(x, 2))
            y_var = tk.DoubleVar(value=round(y, 2))
            self.entry_vars.append((x_var, y_var))

            # Pole X
            x_entry = ttk.Entry(point_frame, textvariable=x_var, width=8)
            x_entry.pack(side=tk.LEFT, padx=5)

            # Pole Y
            y_entry = ttk.Entry(point_frame, textvariable=y_var, width=8)
            y_entry.pack(side=tk.LEFT, padx=5)

            # Przyciski akcji
            button_frame = ttk.Frame(point_frame)
            button_frame.pack(side=tk.LEFT, padx=5)

            ttk.Button(
                button_frame,
                text="✓",
                width=3,
                command=lambda idx=i: self.update_point_from_entry(idx),
            ).pack(side=tk.LEFT)
            ttk.Button(
                button_frame,
                text="×",
                width=3,
                command=lambda idx=i: self.delete_point(idx),
            ).pack(side=tk.LEFT)

            # Bind Enter do aktualizacji
            x_entry.bind("<Return>", lambda e, idx=i: self.update_point_from_entry(idx))
            y_entry.bind("<Return>", lambda e, idx=i: self.update_point_from_entry(idx))

        # Wymuś odświeżenie paska przewijania
        self.points_frame.update_idletasks()
        self.points_frame.master.config(
            scrollregion=self.points_frame.master.bbox("all")
        )

    def update_point_from_entry(self, index):
        """Aktualizacja punktu z pól tekstowych"""
        try:
            if index < len(self.entry_vars):
                x_var, y_var = self.entry_vars[index]
                x = x_var.get()
                y = y_var.get()

                self.control_points[index] = (x, y)
                self.update_curve()
                self.status_var.set(f"Zaktualizowano punkt P{index}")

        except (ValueError, IndexError):
            messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe!")

    def delete_point(self, index):
        """Usuwanie punktu"""
        if 0 <= index < len(self.control_points):
            self.control_points.pop(index)
            self.update_curve()
            self.update_points_display()
            self.status_var.set(f"Usunięto punkt P{index}")

    def add_point(self):
        """Dodawanie nowego punktu"""
        if self.control_points:
            # Dodaj punkt obok ostatniego
            last_point = self.control_points[-1]
            # Użyj bezpiecznych granic
            new_x = min(9.5, last_point[0] + 1)
            new_y = last_point[1]
            new_point = (new_x, new_y)
        else:
            # Pierwszy punkt - środek płótna
            new_point = (5, 5)

        self.control_points.append(new_point)
        self.update_curve()
        self.update_points_display()
        self.status_var.set(f"Dodano punkt P{len(self.control_points)-1}")

    def clear_canvas(self):
        """Czyszczenie płótna"""
        self.control_points = []
        self.dragging_point = None
        self.entry_vars = []
        self.update_curve()
        self.update_points_display()
        self.status_var.set("Płótno wyczyszczone")

    def random_points(self):
        """Generuje losowe punkty kontrolne (Poprawka: użycie aktualnego stopnia)"""
        # A) AKTUALIZACJA STOPNIA Z GUI
        self.degree = self.degree_var.get()

        self.control_points = []
        # B) Liczba punktów = stopień + 1
        num_points = self.degree + 1

        for i in range(num_points):
            # Użycie funkcji losującej z lekkim przesunięciem
            if num_points > 1:
                x = 2 + 6 * i / (num_points - 1)
            else:
                x = 5
            y = np.random.uniform(3, 7)
            self.control_points.append((x, y))

        self.update_curve()
        self.update_points_display()
        self.status_var.set(
            f"Wygenerowano {num_points} losowych punktów (Stopień: {self.degree})"
        )


def main():
    """Uruchomienie aplikacji"""
    root = tk.Tk()
    app = BezierCurveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
