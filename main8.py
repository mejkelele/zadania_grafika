import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import math


class MorphologicalFiltersApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtry Morfologiczne")
        self.root.geometry("1400x900")

        self.image = None
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.structuring_element = None

        self.setup_gui()

    def setup_gui(self):
        """Setup głównego interfejsu GUI"""
        # Główny frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lewy panel - kontrola
        control_frame = ttk.LabelFrame(
            main_frame, text="Operacje Morfologiczne", width=400
        )
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Prawy panel - podgląd obrazów
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Kontrola - ładowanie obrazu
        load_frame = ttk.LabelFrame(control_frame, text="Ładowanie Obrazu")
        load_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            load_frame, text="Wczytaj Obraz PPM/PGM", command=self.load_image
        ).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(load_frame, text="Zapisz Obraz", command=self.save_image).pack(
            fill=tk.X, padx=5, pady=5
        )

        ttk.Button(
            load_frame, text="Konwertuj na Binary", command=self.convert_to_binary
        ).pack(fill=tk.X, padx=5, pady=5)

        # Element strukturyzujący
        se_frame = ttk.LabelFrame(control_frame, text="Element Strukturyzujący")
        se_frame.pack(fill=tk.X, padx=5, pady=5)

        self.se_size = tk.IntVar(value=3)
        size_frame = ttk.Frame(se_frame)
        size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(size_frame, text="Rozmiar:").pack(side=tk.LEFT)
        ttk.Spinbox(
            size_frame, from_=3, to=15, increment=2, textvariable=self.se_size, width=10
        ).pack(side=tk.RIGHT)

        ttk.Button(
            se_frame,
            text="Zdefiniuj Element Strukturyzujący",
            command=self.define_structuring_element,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Operacje morfologiczne
        operations_frame = ttk.LabelFrame(control_frame, text="Operacje Morfologiczne")
        operations_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            operations_frame, text="Dylatacja", command=self.apply_dilation
        ).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(operations_frame, text="Erozja", command=self.apply_erosion).pack(
            fill=tk.X, padx=5, pady=2
        )

        ttk.Button(operations_frame, text="Otwarcie", command=self.apply_opening).pack(
            fill=tk.X, padx=5, pady=2
        )

        ttk.Button(
            operations_frame, text="Domknięcie", command=self.apply_closing
        ).pack(fill=tk.X, padx=5, pady=2)

        # Hit-or-Miss operations
        hitmiss_frame = ttk.LabelFrame(control_frame, text="Hit-or-Miss")
        hitmiss_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            hitmiss_frame, text="Pocienianie (Thinning)", command=self.apply_thinning
        ).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(
            hitmiss_frame,
            text="Pogrubianie (Thickening)",
            command=self.apply_thickening,
        ).pack(fill=tk.X, padx=5, pady=2)

        # Reset
        ttk.Button(
            control_frame, text="Przywróć Oryginał", command=self.reset_image
        ).pack(fill=tk.X, padx=5, pady=10)

        # Podgląd obrazów
        self.setup_preview(preview_frame)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Gotowy")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Domyślny element strukturyzujący
        self.create_default_structuring_element()

    def setup_preview(self, parent):
        """Setup panelu podglądu"""
        # Figura dla obrazów
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Inicjalizacja pustych obrazów
        self.ax1.imshow(np.zeros((100, 100)), cmap="gray")
        self.ax1.set_title("Oryginał")
        self.ax1.axis("off")

        self.ax2.imshow(np.zeros((100, 100)), cmap="gray")
        self.ax2.set_title("Wynik")
        self.ax2.axis("off")

        self.canvas.draw()

    def create_default_structuring_element(self):
        """Tworzy domyślny element strukturyzujący 3x3"""
        size = 3
        self.structuring_element = np.ones((size, size), dtype=np.uint8)
        self.status_var.set("Utworzono domyślny element strukturyzujący 3x3")

    def define_structuring_element(self):
        """Definiowanie własnego elementu strukturyzującego"""
        size = self.se_size.get()

        # Okno do definiowania elementu strukturyzującego
        se_window = tk.Toplevel(self.root)
        se_window.title(f"Element Strukturyzujący {size}x{size}")
        se_window.geometry("400x500")
        se_window.transient(self.root)
        se_window.grab_set()

        entries = []
        checkboxes = []

        # Frame dla przycisków szybkiego wyboru
        quick_frame = ttk.Frame(se_window)
        quick_frame.pack(fill=tk.X, padx=10, pady=5)

        def set_all_ones():
            for row in checkboxes:
                for var in row:
                    var.set(1)

        def set_all_zeros():
            for row in checkboxes:
                for var in row:
                    var.set(0)

        def set_cross():
            center = size // 2
            for i in range(size):
                for j in range(size):
                    if i == center or j == center:
                        checkboxes[i][j].set(1)
                    else:
                        checkboxes[i][j].set(0)

        ttk.Button(quick_frame, text="Wszystkie 1", command=set_all_ones).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(quick_frame, text="Wszystkie 0", command=set_all_zeros).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(quick_frame, text="Krzyż", command=set_cross).pack(
            side=tk.LEFT, padx=2
        )

        # Tworzenie siatki checkboxów
        grid_frame = ttk.Frame(se_window)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for i in range(size):
            row_frame = ttk.Frame(grid_frame)
            row_frame.pack(fill=tk.X, pady=2)

            row_checkboxes = []
            for j in range(size):
                var = tk.IntVar(value=1)
                cb = ttk.Checkbutton(row_frame, variable=var)
                cb.pack(side=tk.LEFT, padx=2)
                row_checkboxes.append(var)
            checkboxes.append(row_checkboxes)

        def apply_structuring_element():
            try:
                kernel = np.zeros((size, size), dtype=np.uint8)
                for i in range(size):
                    for j in range(size):
                        kernel[i, j] = checkboxes[i][j].get()

                self.structuring_element = kernel
                self.status_var.set(
                    f"Zdefiniowano element strukturyzujący {size}x{size}"
                )
                se_window.destroy()

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się utworzyć elementu: {e}")

        ttk.Button(se_window, text="Zastosuj", command=apply_structuring_element).pack(
            pady=10
        )

    def load_image(self):
        """Wczytywanie obrazu PPM/PGM"""
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[
                ("Obrazy PPM/PGM", "*.ppm *.pgm"),
                ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("Wszystkie pliki", "*.*"),
            ],
        )

        if file_path:
            try:
                # Dla PPM/PGM używamy PIL
                pil_image = Image.open(file_path)

                # Konwersja do skali szarości jeśli to kolorowy obraz
                if pil_image.mode != "L":
                    pil_image = pil_image.convert("L")

                self.image = np.array(pil_image)
                self.original_image = self.image.copy()
                self.processed_image = self.image.copy()
                self.image_path = file_path

                self.update_preview()
                self.status_var.set(
                    f"Wczytano: {file_path} - Rozmiar: {self.image.shape}"
                )

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać obrazu: {e}")

    def save_image(self):
        """Zapisywanie obrazu"""
        if self.processed_image is None:
            messagebox.showwarning("Ostrzeżenie", "Brak obrazu do zapisania!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Zapisz obraz jako",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Wszystkie pliki", "*.*")],
        )

        if file_path:
            try:
                pil_image = Image.fromarray(self.processed_image.astype("uint8"))
                pil_image.save(file_path)
                self.status_var.set(f"Zapisano: {file_path}")
                messagebox.showinfo("Sukces", "Obraz zapisany pomyślnie!")

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać obrazu: {e}")

    def convert_to_binary(self):
        """Konwersja obrazu do postaci binarnej"""
        if self.check_image():
            return

        if self.image.max() > 1:  # Jeśli nie jest już binarny
            # Użyj progu Otsu do binaryzacji
            threshold = self.otsu_threshold(self.image)
            self.processed_image = (self.image > threshold).astype(np.uint8) * 255
            self.update_preview()
            self.status_var.set(f"Konwertowano do postaci binarnej (próg: {threshold})")
        else:
            messagebox.showinfo("Info", "Obraz jest już w postaci binarnej")

    def update_preview(self):
        """Aktualizacja podglądu obrazów"""
        if self.image is not None:
            self.ax1.clear()
            self.ax2.clear()

            # Oryginał
            self.ax1.imshow(self.original_image, cmap="gray")
            self.ax1.set_title("Oryginał")
            self.ax1.axis("off")

            # Wynik
            self.ax2.imshow(self.processed_image, cmap="gray")
            self.ax2.set_title("Wynik")
            self.ax2.axis("off")

            self.canvas.draw()

    def check_image(self):
        """Sprawdzenie czy obraz jest wczytany"""
        if self.image is None:
            messagebox.showwarning("Ostrzeżenie", "Najpierw wczytaj obraz!")
            return True
        return False

    def check_binary_image(self):
        """Sprawdzenie czy obraz jest binarny"""
        if self.processed_image is None:
            return True
        unique_vals = np.unique(self.processed_image)
        if len(unique_vals) > 2 or (0 not in unique_vals and 255 not in unique_vals):
            messagebox.showwarning(
                "Ostrzeżenie",
                "Operacja wymaga obrazu binarnego! Użyj 'Konwertuj na Binary'.",
            )
            return True
        return False

    def reset_image(self):
        """Przywracanie oryginalnego obrazu"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_preview()
            self.status_var.set("Przywrócono oryginalny obraz")

    # Podstawowe operacje morfologiczne
    def apply_dilation(self):
        if self.check_image() or self.check_binary_image():
            return
        self.processed_image = self.dilation(
            self.processed_image, self.structuring_element
        )
        self.update_preview()
        self.status_var.set("Zastosowano dylatację")

    def apply_erosion(self):
        if self.check_image() or self.check_binary_image():
            return
        self.processed_image = self.erosion(
            self.processed_image, self.structuring_element
        )
        self.update_preview()
        self.status_var.set("Zastosowano erozję")

    def apply_opening(self):
        if self.check_image() or self.check_binary_image():
            return
        self.processed_image = self.opening(
            self.processed_image, self.structuring_element
        )
        self.update_preview()
        self.status_var.set("Zastosowano otwarcie")

    def apply_closing(self):
        if self.check_image() or self.check_binary_image():
            return
        self.processed_image = self.closing(
            self.processed_image, self.structuring_element
        )
        self.update_preview()
        self.status_var.set("Zastosowano domknięcie")

    def apply_thinning(self):
        if self.check_image() or self.check_binary_image():
            return
        self.processed_image = self.thinning(self.processed_image)
        self.update_preview()
        self.status_var.set("Zastosowano pocienianie")

    def apply_thickening(self):
        if self.check_image() or self.check_binary_image():
            return
        self.processed_image = self.thickening(self.processed_image)
        self.update_preview()
        self.status_var.set("Zastosowano pogrubianie")

    # Implementacje algorytmów morfologicznych
    @staticmethod
    def dilation(image, kernel):
        """Dylatacja - rozszerza jasne obszary"""
        height, width = image.shape
        k_height, k_width = kernel.shape
        pad_h, pad_w = k_height // 2, k_width // 2

        # Dodaj padding
        padded = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
        )
        result = np.zeros_like(image)

        for i in range(height):
            for j in range(width):
                region = padded[i : i + k_height, j : j + k_width]
                # Dylatacja: maksimum z regionu gdzie kernel == 1
                if np.any((kernel == 1) & (region == 255)):
                    result[i, j] = 255
                else:
                    result[i, j] = image[i, j]

        return result

    @staticmethod
    def erosion(image, kernel):
        """Erozja - zmniejsza jasne obszary"""
        height, width = image.shape
        k_height, k_width = kernel.shape
        pad_h, pad_w = k_height // 2, k_width // 2

        # Dodaj padding
        padded = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
        )
        result = np.zeros_like(image)

        for i in range(height):
            for j in range(width):
                region = padded[i : i + k_height, j : j + k_width]
                # Erozja: wszystkie piksele pod kernelem == 1 muszą być białe
                if np.all(region[kernel == 1] == 255):
                    result[i, j] = 255
                else:
                    result[i, j] = 0

        return result

    def opening(self, image, kernel):
        """Otwarcie - erozja followed by dylatacja"""
        eroded = self.erosion(image, kernel)
        return self.dilation(eroded, kernel)

    def closing(self, image, kernel):
        """Domknięcie - dylatacja followed by erozja"""
        dilated = self.dilation(image, kernel)
        return self.erosion(dilated, kernel)

    @staticmethod
    def hit_or_miss(image, kernel_J, kernel_K):
        """Operacja Hit-or-Miss"""
        height, width = image.shape
        k_height, k_width = kernel_J.shape
        pad_h, pad_w = k_height // 2, k_width // 2

        padded = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
        )
        result = np.zeros_like(image)

        for i in range(height):
            for j in range(width):
                region = padded[i : i + k_height, j : j + k_width]

                # Sprawdź warunek Hit: region musi pasować do kernel_J (foreground)
                hit_condition = np.all(region[kernel_J == 1] == 255)

                # Sprawdź warunek Miss: region musi pasować do kernel_K (background)
                miss_condition = np.all(region[kernel_K == 1] == 0)

                if hit_condition and miss_condition:
                    result[i, j] = 255
                else:
                    result[i, j] = image[i, j]

        return result

    def thinning(self, image):
        """Pocienianie przy użyciu operacji Hit-or-Miss"""
        # Zbiór kernelów L dla pocieniania
        L_kernels = [
            (
                np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]], dtype=np.int8),
                np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int8),
            ),
            (
                np.array([[-1, 0, 0], [1, 1, 0], [-1, 1, -1]], dtype=np.int8),
                np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]], dtype=np.int8),
            ),
        ]

        result = image.copy()
        previous = None

        # Iteruj aż do stabilności
        while previous is None or not np.array_equal(result, previous):
            previous = result.copy()

            for kernel_J, kernel_K in L_kernels:
                # Zastosuj operację Hit-or-Miss
                hit_miss_result = self.hit_or_miss(
                    result,
                    (kernel_J == 1).astype(np.uint8),
                    (kernel_K == 1).astype(np.uint8),
                )

                # Odejmij wynik Hit-or-Miss od oryginalnego obrazu
                result = result & ~hit_miss_result

        return result

    def thickening(self, image):
        """Pogrubianie - dopełnienie pocieniania dopełnienia"""
        # Pogrubianie to dopełnienie pocieniania dopełnienia
        complemented = 255 - image
        thinned_complement = self.thinning(complemented)
        return 255 - thinned_complement

    @staticmethod
    def otsu_threshold(image):
        """Automatyczny wybór progu binaryzacji metodą Otsu"""
        # Oblicz histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        # Normalizuj histogram
        hist_norm = hist.astype(float) / hist.sum()

        # Oblicz dystrybuanty i średnie
        omega = np.cumsum(hist_norm)
        mu = np.cumsum(hist_norm * np.arange(256))
        mu_t = mu[-1]

        # Oblicz wariancję międzyklasową
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-10)

        # Znajdź optymalny próg
        optimal_threshold = np.nanargmax(sigma_b_squared)

        return optimal_threshold


def main():
    """Uruchomienie aplikacji"""
    root = tk.Tk()
    app = MorphologicalFiltersApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
