import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import math


class Transformation2DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Przekształcenia 2D - Aplikacja")
        self.root.geometry("1400x900")

        self.figures = []  # Lista figur: {points: [], color: str, name: str}
        self.current_figure = []
        self.dragging_figure = None
        self.dragging_point = None
        self.transform_mode = None  # 'translate', 'rotate', 'scale'
        self.reference_point = None  # Punkt odniesienia dla transformacji
        self.selected_figure = None

        self.setup_gui()

    def setup_gui(self):
        """Setup interfejsu GUI"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lewy panel - kontrola
        control_frame = ttk.LabelFrame(main_frame, text="Kontrola", width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Prawy panel - rysowanie
        draw_frame = ttk.LabelFrame(main_frame, text="Płótno")
        draw_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Rysowanie
        self.setup_canvas(draw_frame)

        # Notebook dla różnych operacji
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Zakładka 1 - Rysowanie figur
        draw_tab = ttk.Frame(notebook)
        notebook.add(draw_tab, text="Rysowanie")
        self.setup_drawing_tab(draw_tab)

        # Zakładka 2 - Przekształcenia
        transform_tab = ttk.Frame(notebook)
        notebook.add(transform_tab, text="Przekształcenia")
        self.setup_transform_tab(transform_tab)

        # Zakładka 3 - Zarządzanie
        manage_tab = ttk.Frame(notebook)
        notebook.add(manage_tab, text="Zarządzanie")
        self.setup_manage_tab(manage_tab)

        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Kliknij na płótnie aby rysować figury")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_canvas(self, parent):
        """Setup płótna do rysowania"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_title("Przekształcenia 2D - Rysuj i przekształcaj figury")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Podpięcie eventów
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def setup_drawing_tab(self, parent):
        """Setup zakładki rysowania"""
        # Nowa figura
        new_fig_frame = ttk.LabelFrame(parent, text="Nowa Figura")
        new_fig_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            new_fig_frame, text="Rozpocznij Nową Figurę", command=self.start_new_figure
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            new_fig_frame, text="Zakończ Figurę", command=self.finish_figure
        ).pack(fill=tk.X, padx=5, pady=2)

        # Punkty bieżącej figury
        self.points_frame = ttk.LabelFrame(parent, text="Punkty Bieżącej Figury")
        self.points_frame.pack(fill=tk.X, padx=5, pady=5)

        # Lista figur
        self.figures_listbox = tk.Listbox(parent, height=8)
        self.figures_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.figures_listbox.bind("<<ListboxSelect>>", self.on_figure_select)

        ttk.Button(
            parent, text="Usuń Zaznaczoną Figurę", command=self.delete_selected_figure
        ).pack(fill=tk.X, padx=5, pady=2)

    def setup_transform_tab(self, parent):
        """Setup zakładki przekształceń"""
        # Wybór trybu
        mode_frame = ttk.LabelFrame(parent, text="Tryb Przekształcenia")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            mode_frame,
            text="Przesunięcie",
            command=lambda: self.set_transform_mode("translate"),
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            mode_frame, text="Obrót", command=lambda: self.set_transform_mode("rotate")
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            mode_frame,
            text="Skalowanie",
            command=lambda: self.set_transform_mode("scale"),
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            mode_frame, text="Anuluj Tryb", command=self.cancel_transform_mode
        ).pack(fill=tk.X, padx=5, pady=2)

        # Przesunięcie numeryczne
        translate_frame = ttk.LabelFrame(parent, text="Przesunięcie Numeryczne")
        translate_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(translate_frame, text="X:").pack(side=tk.LEFT, padx=2)
        self.translate_x = ttk.Entry(translate_frame, width=6)
        self.translate_x.pack(side=tk.LEFT, padx=2)

        ttk.Label(translate_frame, text="Y:").pack(side=tk.LEFT, padx=2)
        self.translate_y = ttk.Entry(translate_frame, width=6)
        self.translate_y.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            translate_frame, text="Przesuń", command=self.numeric_translate
        ).pack(side=tk.LEFT, padx=5)

        # Obrót numeryczny
        rotate_frame = ttk.LabelFrame(parent, text="Obrót Numeryczny")
        rotate_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(rotate_frame, text="Punkt X:").pack(side=tk.LEFT, padx=2)
        self.rotate_center_x = ttk.Entry(rotate_frame, width=6)
        self.rotate_center_x.pack(side=tk.LEFT, padx=2)

        ttk.Label(rotate_frame, text="Y:").pack(side=tk.LEFT, padx=2)
        self.rotate_center_y = ttk.Entry(rotate_frame, width=6)
        self.rotate_center_y.pack(side=tk.LEFT, padx=2)

        ttk.Label(rotate_frame, text="Kąt:").pack(side=tk.LEFT, padx=2)
        self.rotate_angle = ttk.Entry(rotate_frame, width=6)
        self.rotate_angle.pack(side=tk.LEFT, padx=2)

        ttk.Button(rotate_frame, text="Obróć", command=self.numeric_rotate).pack(
            side=tk.LEFT, padx=5
        )

        # Skalowanie numeryczne
        scale_frame = ttk.LabelFrame(parent, text="Skalowanie Numeryczne")
        scale_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(scale_frame, text="Punkt X:").pack(side=tk.LEFT, padx=2)
        self.scale_center_x = ttk.Entry(scale_frame, width=6)
        self.scale_center_x.pack(side=tk.LEFT, padx=2)

        ttk.Label(scale_frame, text="Y:").pack(side=tk.LEFT, padx=2)
        self.scale_center_y = ttk.Entry(scale_frame, width=6)
        self.scale_center_y.pack(side=tk.LEFT, padx=2)

        ttk.Label(scale_frame, text="Skala:").pack(side=tk.LEFT, padx=2)
        self.scale_factor = ttk.Entry(scale_frame, width=6)
        self.scale_factor.pack(side=tk.LEFT, padx=2)

        ttk.Button(scale_frame, text="Skaluj", command=self.numeric_scale).pack(
            side=tk.LEFT, padx=5
        )

    def setup_manage_tab(self, parent):
        """Setup zakładki zarządzania"""
        ttk.Button(parent, text="Zapisz Scenę", command=self.save_scene).pack(
            fill=tk.X, padx=5, pady=5
        )
        ttk.Button(parent, text="Wczytaj Scenę", command=self.load_scene).pack(
            fill=tk.X, padx=5, pady=5
        )
        ttk.Button(parent, text="Wyczyść Wszystko", command=self.clear_all).pack(
            fill=tk.X, padx=5, pady=5
        )

        # Informacje
        info_frame = ttk.LabelFrame(parent, text="Instrukcje")
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        info_text = """
1. Rysowanie: Klikaj na płótnie
2. Zakończ figurę przyciskiem
3. Wybierz figurę z listy
4. Przeciągaj myszą lub użyj pól
5. Zapisz/wczytaj scenę
        """
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(padx=5, pady=5)

    # Macierze transformacji (współrzędne jednorodne)
    def translation_matrix(self, dx, dy):
        """Macierz translacji"""
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    def rotation_matrix(self, angle, center=(0, 0)):
        """Macierz obrotu wokół punktu"""
        cx, cy = center
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Translacja do punktu obrotu -> obrót -> translacja z powrotem
        T1 = self.translation_matrix(-cx, -cy)
        R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        T2 = self.translation_matrix(cx, cy)

        return T2 @ R @ T1

    def scaling_matrix(self, sx, sy, center=(0, 0)):
        """Macierz skalowania względem punktu"""
        cx, cy = center

        # Translacja do punktu skalowania -> skalowanie -> translacja z powrotem
        T1 = self.translation_matrix(-cx, -cy)
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        T2 = self.translation_matrix(cx, cy)

        return T2 @ S @ T1

    def apply_transform_to_points(self, points, transform_matrix):
        """Zastosuj transformację do listy punktów"""
        transformed_points = []
        for x, y in points:
            # Konwersja do współrzędnych jednorodnych
            point_homogeneous = np.array([x, y, 1])
            transformed = transform_matrix @ point_homogeneous
            transformed_points.append((transformed[0], transformed[1]))
        return transformed_points

    def draw_scene(self):
        """Rysuje całą scenę"""
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_title("Przekształcenia 2D - Rysuj i przekształcaj figury")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)

        # Rysuj wszystkie figury
        colors = ["red", "blue", "green", "orange", "purple", "brown"]
        for i, figure in enumerate(self.figures):
            color = figure.get("color", colors[i % len(colors)])
            points = figure["points"]

            if len(points) >= 3:
                polygon = Polygon(
                    points, closed=True, alpha=0.7, color=color, picker=True
                )
                self.ax.add_patch(polygon)

            # Rysuj punkty kontrolne
            for j, (x, y) in enumerate(points):
                self.ax.plot(x, y, "o", color="darkred", markersize=6)
                self.ax.text(x + 0.2, y + 0.2, f"{j}", fontsize=8, color="darkred")

        # Rysuj bieżącą figurę
        if len(self.current_figure) >= 2:
            points_array = np.array(self.current_figure)
            self.ax.plot(points_array[:, 0], points_array[:, 1], "ro-", alpha=0.5)
        elif len(self.current_figure) == 1:
            x, y = self.current_figure[0]
            self.ax.plot(x, y, "ro", markersize=8)

        # Rysuj punkt odniesienia jeśli istnieje
        if self.reference_point:
            rx, ry = self.reference_point
            self.ax.plot(
                rx, ry, "s", color="gold", markersize=10, markeredgecolor="black"
            )

        self.canvas.draw()
        self.update_figures_list()

    def update_figures_list(self):
        """Aktualizuje listę figur"""
        self.figures_listbox.delete(0, tk.END)
        for i, figure in enumerate(self.figures):
            name = figure.get("name", f"Figura {i+1}")
            points_count = len(figure["points"])
            self.figures_listbox.insert(tk.END, f"{name} ({points_count} punktów)")

    def on_click(self, event):
        """Obsługa kliknięcia myszą"""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        # Tryb transformacji
        if self.transform_mode and self.selected_figure is not None:
            if self.transform_mode == "translate":
                self.start_translation((x, y))
            elif self.transform_mode == "rotate":
                if self.reference_point is None:
                    self.reference_point = (x, y)
                    self.status_var.set(f"Ustawiono punkt obrotu: ({x:.1f}, {y:.1f})")
                else:
                    self.start_rotation((x, y))
            elif self.transform_mode == "scale":
                if self.reference_point is None:
                    self.reference_point = (x, y)
                    self.status_var.set(
                        f"Ustawiono punkt skalowania: ({x:.1f}, {y:.1f})"
                    )
                else:
                    self.start_scaling((x, y))
            return

        # Rysowanie nowej figury
        self.current_figure.append((x, y))
        self.status_var.set(f"Dodano punkt: ({x:.1f}, {y:.1f})")
        self.draw_scene()

    def on_motion(self, event):
        """Obsługa ruchu myszą"""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        # Tutaj można dodać podgląd podczas przeciągania
        self.draw_scene()

    def on_release(self, event):
        """Obsługa zwolnienia przycisku myszy"""
        pass

    def on_figure_select(self, event):
        """Obsługa wyboru figury z listy"""
        selection = self.figures_listbox.curselection()
        if selection:
            self.selected_figure = selection[0]
            self.status_var.set(f"Wybrano figurę {self.selected_figure + 1}")

    def set_transform_mode(self, mode):
        """Ustawia tryb transformacji"""
        self.transform_mode = mode
        if self.selected_figure is None:
            messagebox.showwarning("Uwaga", "Najpierw wybierz figurę z listy!")
            self.transform_mode = None
            return

        self.reference_point = None
        if mode == "translate":
            self.status_var.set("Tryb przesunięcia - przeciągnij figurę")
        elif mode == "rotate":
            self.status_var.set("Tryb obrotu - najpierw kliknij punkt obrotu")
        elif mode == "scale":
            self.status_var.set("Tryb skalowania - najpierw kliknij punkt skalowania")

    def cancel_transform_mode(self):
        """Anuluje tryb transformacji"""
        self.transform_mode = None
        self.reference_point = None
        self.status_var.set("Anulowano tryb transformacji")

    def start_translation(self, start_point):
        """Rozpoczyna przesunięcie"""
        if self.selected_figure is not None:
            figure = self.figures[self.selected_figure]
            # Tutaj można dodać logikę przeciągania dla przesunięcia
            self.status_var.set("Przeciągaj figurę do nowej pozycji")

    def start_rotation(self, start_point):
        """Rozpoczyna obrót"""
        # Tutaj można dodać logikę przeciągania dla obrotu
        self.status_var.set("Przeciągaj aby obrócić figurę")

    def start_scaling(self, start_point):
        """Rozpoczyna skalowanie"""
        # Tutaj można dodać logikę przeciągania dla skalowania
        self.status_var.set("Przeciągaj aby przeskalować figurę")

    # Operacje numeryczne
    def numeric_translate(self):
        """Przesunięcie numeryczne"""
        if self.selected_figure is None:
            messagebox.showwarning("Uwaga", "Najpierw wybierz figurę!")
            return

        try:
            dx = float(self.translate_x.get())
            dy = float(self.translate_y.get())

            figure = self.figures[self.selected_figure]
            transform_matrix = self.translation_matrix(dx, dy)
            figure["points"] = self.apply_transform_to_points(
                figure["points"], transform_matrix
            )

            self.draw_scene()
            self.status_var.set(f"Przesunięto o wektor ({dx}, {dy})")

        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe!")

    def numeric_rotate(self):
        """Obrót numeryczny"""
        if self.selected_figure is None:
            messagebox.showwarning("Uwaga", "Najpierw wybierz figurę!")
            return

        try:
            cx = float(self.rotate_center_x.get())
            cy = float(self.rotate_center_y.get())
            angle = float(self.rotate_angle.get())

            figure = self.figures[self.selected_figure]
            transform_matrix = self.rotation_matrix(angle, (cx, cy))
            figure["points"] = self.apply_transform_to_points(
                figure["points"], transform_matrix
            )

            self.draw_scene()
            self.status_var.set(f"Obrócono o {angle}° wokół ({cx}, {cy})")

        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe!")

    def numeric_scale(self):
        """Skalowanie numeryczne"""
        if self.selected_figure is None:
            messagebox.showwarning("Uwaga", "Najpierw wybierz figurę!")
            return

        try:
            cx = float(self.scale_center_x.get())
            cy = float(self.scale_center_y.get())
            scale = float(self.scale_factor.get())

            figure = self.figures[self.selected_figure]
            transform_matrix = self.scaling_matrix(scale, scale, (cx, cy))
            figure["points"] = self.apply_transform_to_points(
                figure["points"], transform_matrix
            )

            self.draw_scene()
            self.status_var.set(f"Przeskalowano x{scale} względem ({cx}, {cy})")

        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe!")

    # Zarządzanie figurami
    def start_new_figure(self):
        """Rozpoczyna nową figurę"""
        self.current_figure = []
        self.status_var.set("Rozpoczęto nową figurę - klikaj punkty na płótnie")

    def finish_figure(self):
        """Kończy bieżącą figurę"""
        if len(self.current_figure) >= 3:
            self.figures.append(
                {
                    "points": self.current_figure.copy(),
                    "color": f"C{len(self.figures)}",
                    "name": f"Figura {len(self.figures) + 1}",
                }
            )
            self.current_figure = []
            self.draw_scene()
            self.status_var.set("Dodano nową figurę")
        else:
            messagebox.showwarning("Uwaga", "Figura musi mieć co najmniej 3 punkty!")

    def delete_selected_figure(self):
        """Usuwa zaznaczoną figurę"""
        if self.selected_figure is not None:
            self.figures.pop(self.selected_figure)
            self.selected_figure = None
            self.draw_scene()
            self.status_var.set("Usunięto figurę")

    def clear_all(self):
        """Czyści wszystko"""
        self.figures = []
        self.current_figure = []
        self.selected_figure = None
        self.transform_mode = None
        self.reference_point = None
        self.draw_scene()
        self.status_var.set("Wyczyszczono scenę")

    # Serializacja
    def save_scene(self):
        """Zapisuje scenę do pliku"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            try:
                # Przygotuj dane do zapisu
                scene_data = {
                    "figures": self.figures,
                    "current_figure": self.current_figure,
                }

                with open(file_path, "w") as f:
                    json.dump(scene_data, f, indent=2)

                self.status_var.set(f"Zapisano scenę do: {file_path}")
                messagebox.showinfo("Sukces", "Scena zapisana pomyślnie!")

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać: {e}")

    def load_scene(self):
        """Wczytuje scenę z pliku"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, "r") as f:
                    scene_data = json.load(f)

                self.figures = scene_data.get("figures", [])
                self.current_figure = scene_data.get("current_figure", [])
                self.selected_figure = None
                self.transform_mode = None
                self.reference_point = None

                self.draw_scene()
                self.status_var.set(f"Wczytano scenę z: {file_path}")
                messagebox.showinfo("Sukces", "Scena wczytana pomyślnie!")

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać: {e}")


def main():
    """Uruchomienie aplikacji"""
    root = tk.Tk()
    app = Transformation2DApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
