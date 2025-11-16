import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class GreenAreaAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Analiza Terenów Zielonych - Zdjęcia Satelitarne")
        self.root.geometry("1200x800")

        self.image = None
        self.original_image = None
        self.green_mask = None
        self.green_percentage = 0

        self.setup_gui()

    def setup_gui(self):
        """Setup interfejsu GUI"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lewy panel - kontrola
        control_frame = ttk.LabelFrame(main_frame, text="Kontrola", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Prawy panel - podgląd
        preview_frame = ttk.LabelFrame(main_frame, text="Analiza")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Przyciski
        ttk.Button(
            control_frame, text="Wczytaj Zdjęcie Satelitarne", command=self.load_image
        ).pack(fill=tk.X, padx=5, pady=5)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=10)

        # Metody analizy
        method_frame = ttk.LabelFrame(control_frame, text="Metoda Analizy")
        method_frame.pack(fill=tk.X, padx=5, pady=5)

        self.method_var = tk.StringVar(value="ndvi")

        ttk.Radiobutton(
            method_frame,
            text="NDVI (Normalized Difference Vegetation Index)",
            variable=self.method_var,
            value="ndvi",
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            method_frame,
            text="Kolor RGB - zakres zieleni",
            variable=self.method_var,
            value="rgb",
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            method_frame,
            text="Prosty wykrywacz zieleni",
            variable=self.method_var,
            value="simple",
        ).pack(anchor="w", pady=2)

        ttk.Button(
            control_frame,
            text="Analizuj Tereny Zielone",
            command=self.analyze_green_areas,
        ).pack(fill=tk.X, padx=5, pady=10)

        # Wyniki
        result_frame = ttk.LabelFrame(control_frame, text="Wyniki")
        result_frame.pack(fill=tk.X, padx=5, pady=5)

        self.result_var = tk.StringVar()
        self.result_var.set("Kliknij 'Analizuj' aby obliczyć")
        ttk.Label(result_frame, textvariable=self.result_var, wraplength=280).pack(
            padx=5, pady=5
        )

        # Ustawienia
        settings_frame = ttk.LabelFrame(control_frame, text="Ustawienia")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        # Próg dla NDVI
        ttk.Label(settings_frame, text="Próg zieleni:").pack(anchor="w")
        self.threshold_var = tk.DoubleVar(value=0.2)
        ttk.Scale(
            settings_frame,
            from_=0.0,
            to=0.5,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, pady=2)

        # Podgląd
        self.setup_preview(preview_frame)

    def setup_preview(self, parent):
        """Setup panelu podglądu"""
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(
            2, 2, figsize=(10, 8)
        )
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Inicjalizacja pustych obrazów
        for ax, title in zip(
            [self.ax1, self.ax2, self.ax3, self.ax4],
            ["Oryginał", "Maska Zieleni", "Wynik", "Statystyki"],
        ):
            if ax != self.ax4:
                ax.imshow(np.zeros((100, 100, 3)))
            ax.set_title(title)
            ax.axis("off")

        self.canvas.draw()

    def load_image(self):
        """Wczytuje zdjęcie satelitarne"""
        file_path = filedialog.askopenfilename(
            title="Wybierz zdjęcie satelitarne",
            filetypes=[
                ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("Wszystkie", "*.*"),
            ],
        )

        if file_path:
            try:
                # Wczytaj obraz w kolorze (RGB)
                img = Image.open(file_path)
                self.original_image = np.array(img)

                # Jeśli obraz ma kanał alpha, usuń go
                if self.original_image.shape[2] == 4:
                    self.original_image = self.original_image[:, :, :3]

                self.image = self.original_image.copy()
                self.update_display()

                messagebox.showinfo("Sukces", "Zdjęcie satelitarne wczytane pomyślnie!")

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać: {str(e)}")

    def calculate_ndvi(self, image):
        """Oblicza NDVI - Normalized Difference Vegetation Index"""
        # Konwersja do float dla precyzyjnych obliczeń
        img_float = image.astype(np.float32) / 255.0

        # Dla standardowych zdjęć RGB (nie multispektralnych)
        # Używamy: Red = czerwony, Green = zielony (jako przybliżenie NIR)
        if image.shape[2] >= 3:
            red = img_float[:, :, 0]  # Czerwony
            green = img_float[:, :, 1]  # Zielony (przybliżenie NIR)

            # Oblicz NDVI: (GREEN - RED) / (GREEN + RED)
            ndvi = (green - red) / (
                green + red + 1e-8
            )  # +1e-8 aby uniknąć dzielenia przez 0

            # Normalizacja do zakresu -1 do 1
            ndvi = np.clip(ndvi, -1, 1)
            return ndvi

        return None

    def rgb_green_detection(self, image):
        """Wykrywanie zieleni w przestrzeni RGB"""
        img_float = image.astype(np.float32)

        # Ekstrakcja kanałów
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]

        # Warunki dla zieleni w RGB
        # Zielony > Czerwony i Zielony > Niebieski
        green_condition = (g > r) & (g > b) & (g > 50)

        return green_condition.astype(np.float32)

    def simple_green_detection(self, image):
        """Proste wykrywanie zieleni - tylko zielony kanał"""
        green_channel = image[:, :, 1].astype(np.float32)

        # Normalizuj i zastosuj próg
        green_normalized = green_channel / 255.0
        threshold = self.threshold_var.get()

        return (green_normalized > threshold).astype(np.float32)

    def analyze_green_areas(self):
        """Główna funkcja analizy terenów zielonych"""
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj zdjęcie satelitarne!")
            return

        try:
            method = self.method_var.get()
            green_mask = None

            if method == "ndvi":
                green_mask = self.calculate_ndvi(self.original_image)
                threshold = self.threshold_var.get()
                green_binary = (green_mask > threshold).astype(np.uint8)

            elif method == "rgb":
                green_mask = self.rgb_green_detection(self.original_image)
                green_binary = (green_mask > 0.5).astype(np.uint8)

            elif method == "simple":
                green_mask = self.simple_green_detection(self.original_image)
                green_binary = (green_mask > 0.5).astype(np.uint8)

            if green_mask is not None:
                # Oblicz procent terenów zielonych
                total_pixels = green_binary.size
                green_pixels = np.sum(green_binary)
                self.green_percentage = (green_pixels / total_pixels) * 100

                self.green_mask = green_binary
                self.update_results()

            else:
                messagebox.showerror("Błąd", "Nie udało się obliczyć maski zieleni")

        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas analizy: {str(e)}")

    def update_results(self):
        """Aktualizuje wyniki i wyświetla je"""
        # Aktualizuj tekst wyników
        result_text = f"Tereny zielone: {self.green_percentage:.2f}%\n"
        result_text += f"Metoda: {self.method_var.get().upper()}\n"
        result_text += f"Próg: {self.threshold_var.get():.2f}\n"

        if self.green_percentage < 10:
            result_text += "Bardzo mało zieleni (teren miejski/industrialny)"
        elif self.green_percentage < 30:
            result_text += "Umiarkowana ilość zieleni"
        elif self.green_percentage < 60:
            result_text += "Dużo terenów zielonych"
        else:
            result_text += "Bardzo dużo terenów zielonych (las/park)"

        self.result_var.set(result_text)

        # Aktualizuj wyświetlanie
        self.update_display()

    def update_display(self):
        """Aktualizuje podgląd obrazów"""
        if self.original_image is not None:
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()

            # Oryginał
            self.ax1.imshow(self.original_image)
            self.ax1.set_title("Zdjęcie Satelitarne")
            self.ax1.axis("off")

            # Maska zieleni
            if self.green_mask is not None:
                self.ax2.imshow(self.green_mask, cmap="Greens")
                self.ax2.set_title("Maska Terenów Zielonych")
                self.ax2.axis("off")

                # Wynik - nałożenie maski na oryginał
                result_image = self.original_image.copy()
                # Podświetl zieleń na czerwono (z przezroczystością)
                highlight = self.green_mask == 1
                result_image[highlight] = [255, 0, 0]  # Czerwony dla zieleni

                self.ax3.imshow(result_image)
                self.ax3.set_title("Tereny Zielone (czerwone)")
                self.ax3.axis("off")

                # Statystyki
                self.ax4.text(
                    0.1,
                    0.9,
                    f"Tereny zielone: {self.green_percentage:.2f}%",
                    transform=self.ax4.transAxes,
                    fontsize=12,
                    fontweight="bold",
                )
                self.ax4.text(
                    0.1,
                    0.7,
                    f"Metoda: {self.method_var.get().upper()}",
                    transform=self.ax4.transAxes,
                    fontsize=10,
                )
                self.ax4.text(
                    0.1,
                    0.5,
                    f"Próg: {self.threshold_var.get():.2f}",
                    transform=self.ax4.transAxes,
                    fontsize=10,
                )

                total_pixels = self.green_mask.size
                green_pixels = np.sum(self.green_mask)
                self.ax4.text(
                    0.1,
                    0.3,
                    f"Piksele zielone: {green_pixels:,}",
                    transform=self.ax4.transAxes,
                    fontsize=10,
                )
                self.ax4.text(
                    0.1,
                    0.1,
                    f"Wszystkie piksele: {total_pixels:,}",
                    transform=self.ax4.transAxes,
                    fontsize=10,
                )

                self.ax4.set_xlim(0, 1)
                self.ax4.set_ylim(0, 1)
                self.ax4.axis("off")

            self.canvas.draw()


# Uruchomienie
if __name__ == "__main__":
    root = tk.Tk()
    app = GreenAreaAnalyzer(root)
    root.mainloop()
