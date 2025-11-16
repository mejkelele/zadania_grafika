import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy import ndimage
import math
from collections import Counter


class ImageEnhancementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Polepszanie Jakości Obrazów - Histogram i Binaryzacja")
        self.root.geometry("1400x900")

        self.image = None
        self.original_image = None
        self.processed_image = None
        self.image_path = None

        self.setup_gui()

    def setup_gui(self):
        """Setup głównego interfejsu GUI"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lewy panel - kontrola
        control_frame = ttk.LabelFrame(main_frame, text="Operacje", width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Prawy panel - podgląd obrazów i histogramów
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Kontrola - ładowanie obrazu
        load_frame = ttk.LabelFrame(control_frame, text="Ładowanie Obrazu")
        load_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(load_frame, text="Wczytaj Obraz", command=self.load_image).pack(
            fill=tk.X, padx=5, pady=5
        )
        ttk.Button(load_frame, text="Zapisz Obraz", command=self.save_image).pack(
            fill=tk.X, padx=5, pady=5
        )

        # Notebook dla różnych operacji
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

        # Zakładka 1 - Filtry
        filter_frame = ttk.Frame(notebook)
        notebook.add(filter_frame, text="Filtry")
        self.setup_filters(filter_frame)

        # Zakładka 2 - Histogram
        histogram_frame = ttk.Frame(notebook)
        notebook.add(histogram_frame, text="Histogram")
        self.setup_histogram_operations(histogram_frame)

        # Zakładka 3 - Binaryzacja
        binarization_frame = ttk.Frame(notebook)
        notebook.add(binarization_frame, text="Binaryzacja")
        self.setup_binarization_operations(binarization_frame)

        # Podgląd obrazów i histogramów
        self.setup_preview(preview_frame)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Gotowy")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_filters(self, parent):
        """Setup interfejsu filtrów"""
        # Filtr uśredniający
        avg_frame = ttk.LabelFrame(parent, text="Filtr Uśredniający")
        avg_frame.pack(fill=tk.X, padx=5, pady=5)

        self.avg_size = tk.IntVar(value=3)
        ttk.Spinbox(
            avg_frame, from_=3, to=15, increment=2, textvariable=self.avg_size
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            avg_frame,
            text="Zastosuj Filtr Uśredniający",
            command=self.apply_average_filter,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Filtr medianowy
        med_frame = ttk.LabelFrame(parent, text="Filtr Medianowy")
        med_frame.pack(fill=tk.X, padx=5, pady=5)

        self.med_size = tk.IntVar(value=3)
        ttk.Spinbox(
            med_frame, from_=3, to=15, increment=2, textvariable=self.med_size
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            med_frame, text="Zastosuj Filtr Medianowy", command=self.apply_median_filter
        ).pack(fill=tk.X, padx=5, pady=5)

        # Filtr Sobel
        sobel_frame = ttk.LabelFrame(parent, text="Wykrywanie Krawędzi")
        sobel_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            sobel_frame, text="Filtr Sobel", command=self.apply_sobel_filter
        ).pack(fill=tk.X, padx=5, pady=5)

        # Filtr górnoprzepustowy
        highpass_frame = ttk.LabelFrame(parent, text="Wyostrzanie")
        highpass_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            highpass_frame,
            text="Filtr Górnoprzepustowy",
            command=self.apply_highpass_filter,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Rozmycie Gaussa
        gauss_frame = ttk.LabelFrame(parent, text="Rozmycie Gaussa")
        gauss_frame.pack(fill=tk.X, padx=5, pady=5)

        self.sigma_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            gauss_frame,
            from_=0.1,
            to=5.0,
            variable=self.sigma_var,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(gauss_frame, text="Sigma: 1.0").pack()
        ttk.Button(
            gauss_frame,
            text="Zastosuj Rozmycie Gaussa",
            command=self.apply_gaussian_blur,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Własna maska
        custom_frame = ttk.LabelFrame(parent, text="Własna Maska")
        custom_frame.pack(fill=tk.X, padx=5, pady=5)

        self.custom_size = tk.IntVar(value=3)
        ttk.Spinbox(
            custom_frame, from_=3, to=9, increment=2, textvariable=self.custom_size
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            custom_frame, text="Zdefiniuj Maskę", command=self.define_custom_mask
        ).pack(fill=tk.X, padx=5, pady=5)

    def setup_histogram_operations(self, parent):
        """Setup operacji na histogramie"""
        # Rozszerzenie histogramu
        stretch_frame = ttk.LabelFrame(parent, text="Rozszerzenie Histogramu")
        stretch_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            stretch_frame,
            text="Rozszerz Histogram",
            command=self.apply_histogram_stretching,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Wyrównanie histogramu
        equalize_frame = ttk.LabelFrame(parent, text="Wyrównanie Histogramu")
        equalize_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            equalize_frame,
            text="Wyrównaj Histogram",
            command=self.apply_histogram_equalization,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Wyświetlanie histogramu
        hist_frame = ttk.LabelFrame(parent, text="Wyświetl Histogram")
        hist_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            hist_frame,
            text="Pokaż Histogram Oryginału",
            command=self.show_original_histogram,
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            hist_frame,
            text="Pokaż Histogram Wyniku",
            command=self.show_processed_histogram,
        ).pack(fill=tk.X, padx=5, pady=2)

    def setup_binarization_operations(self, parent):
        """Setup operacji binaryzacji"""
        # Binaryzacja ręczna
        manual_frame = ttk.LabelFrame(parent, text="Binaryzacja Ręczna")
        manual_frame.pack(fill=tk.X, padx=5, pady=5)

        self.manual_threshold = tk.IntVar(value=128)
        threshold_scale = ttk.Scale(
            manual_frame,
            from_=0,
            to=255,
            variable=self.manual_threshold,
            orient=tk.HORIZONTAL,
        )
        threshold_scale.pack(fill=tk.X, padx=5, pady=2)

        # POPRAWKA GUI: Dynamiczna etykieta
        self.manual_threshold_label = ttk.Label(
            manual_frame, text=f"Próg: {self.manual_threshold.get()}"
        )
        self.manual_threshold_label.pack()
        threshold_scale.bind("<Motion>", self.update_manual_threshold_label)
        threshold_scale.bind("<ButtonRelease-1>", self.update_manual_threshold_label)

        ttk.Button(
            manual_frame,
            text="Binaryzacja Ręczna",
            command=self.apply_manual_binarization,
        ).pack(fill=tk.X, padx=5, pady=5)

        # Metody automatyczne
        auto_frame = ttk.LabelFrame(parent, text="Binaryzacja Automatyczna")
        auto_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            auto_frame, text="Percent Black Selection", command=self.apply_percent_black
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            auto_frame,
            text="Mean Iterative Selection",
            command=self.apply_mean_iterative,
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            auto_frame, text="Entropy Selection", command=self.apply_entropy_selection
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            auto_frame, text="Minimum Error", command=self.apply_minimum_error
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            auto_frame,
            text="Fuzzy Minimum Error",
            command=self.apply_fuzzy_minimum_error,
        ).pack(fill=tk.X, padx=5, pady=2)

    def update_manual_threshold_label(self, event=None):
        """Aktualizuje etykietę progu manualnego."""
        self.manual_threshold_label.config(text=f"Próg: {self.manual_threshold.get()}")

    def setup_preview(self, parent):
        """Setup panelu podglądu"""
        # Notebook dla obrazów i histogramów
        preview_notebook = ttk.Notebook(parent)
        preview_notebook.pack(fill=tk.BOTH, expand=True)

        # Zakładka z obrazami
        image_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(image_frame, text="Obrazy")

        # Figura dla obrazów
        self.fig_images, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig_images.tight_layout(pad=3.0)

        self.canvas_images = FigureCanvasTkAgg(self.fig_images, image_frame)
        self.canvas_images.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Inicjalizacja pustych obrazów
        self.ax1.imshow(np.zeros((100, 100, 3)), cmap="gray")
        self.ax1.set_title("Oryginał")
        self.ax1.axis("off")

        self.ax2.imshow(np.zeros((100, 100, 3)), cmap="gray")
        self.ax2.set_title("Wynik")
        self.ax2.axis("off")

        # Zakładka z histogramami
        hist_frame = ttk.Frame(preview_notebook)
        preview_notebook.add(hist_frame, text="Histogramy")

        # Figura dla histogramów
        self.fig_hist, (self.ax_hist1, self.ax_hist2) = plt.subplots(
            1, 2, figsize=(10, 4)
        )
        self.fig_hist.tight_layout(pad=3.0)

        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, hist_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Inicjalizacja pustych histogramów
        self.ax_hist1.bar(range(256), [0] * 256, color="gray", alpha=0.7)
        self.ax_hist1.set_title("Histogram Oryginału")
        self.ax_hist1.set_xlim(0, 255)

        self.ax_hist2.bar(range(256), [0] * 256, color="blue", alpha=0.7)
        self.ax_hist2.set_title("Histogram Wyniku")
        self.ax_hist2.set_xlim(0, 255)

        self.canvas_images.draw()
        self.canvas_hist.draw()

    def load_image(self):
        """Wczytywanie obrazu"""
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[
                ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.ppm *.pgm"),
                ("Wszystkie pliki", "*.*"),
            ],
        )

        if file_path:
            try:
                pil_image = Image.open(file_path)
                self.image = np.array(pil_image)
                self.original_image = self.image.copy()
                self.processed_image = self.image.copy()
                self.image_path = file_path

                self.update_preview()
                self.update_histograms()
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

    def update_preview(self):
        """Aktualizacja podglądu obrazów"""
        if self.image is not None:
            self.ax1.clear()
            self.ax2.clear()

            # Oryginał
            if len(self.original_image.shape) == 2:
                self.ax1.imshow(self.original_image, cmap="gray")
            else:
                self.ax1.imshow(self.original_image)
            self.ax1.set_title("Oryginał")
            self.ax1.axis("off")

            # Wynik
            if len(self.processed_image.shape) == 2:
                self.ax2.imshow(self.processed_image, cmap="gray")
            else:
                self.ax2.imshow(self.processed_image)
            self.ax2.set_title("Wynik")
            self.ax2.axis("off")

            self.canvas_images.draw()

    def update_histograms(self):
        """Aktualizacja histogramów"""
        if self.image is not None:
            self.ax_hist1.clear()
            self.ax_hist2.clear()

            # Histogram oryginału
            if len(self.original_image.shape) == 2:
                hist_orig, bins = np.histogram(
                    self.original_image.flatten(), 256, [0, 256]
                )
            else:
                gray_orig = self.rgb_to_grayscale(self.original_image)
                hist_orig, bins = np.histogram(gray_orig.flatten(), 256, [0, 256])

            self.ax_hist1.bar(bins[:-1], hist_orig, color="gray", alpha=0.7)
            self.ax_hist1.set_title("Histogram Oryginału")
            self.ax_hist1.set_xlim(0, 255)

            # Histogram wyniku
            if len(self.processed_image.shape) == 2:
                hist_proc, bins = np.histogram(
                    self.processed_image.flatten(), 256, [0, 256]
                )
            else:
                gray_proc = self.rgb_to_grayscale(self.processed_image)
                hist_proc, bins = np.histogram(gray_proc.flatten(), 256, [0, 256])

            self.ax_hist2.bar(bins[:-1], hist_proc, color="blue", alpha=0.7)
            self.ax_hist2.set_title("Histogram Wyniku")
            self.ax_hist2.set_xlim(0, 255)

            self.canvas_hist.draw()

    def show_original_histogram(self):
        """Pokazanie histogramu oryginalnego obrazu"""
        if self.check_image():
            return
        self.update_histograms()

    def show_processed_histogram(self):
        """Pokazanie histogramu przetworzonego obrazu"""
        if self.check_image():
            return
        self.update_histograms()

    # Metody filtrów
    def apply_average_filter(self):
        if self.check_image():
            return
        size = self.avg_size.get()
        self.processed_image = self.average_filter(self.original_image, size)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Zastosowano filtr uśredniający {size}x{size}")

    def apply_median_filter(self):
        if self.check_image():
            return
        size = self.med_size.get()
        self.processed_image = self.median_filter(self.original_image, size)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Zastosowano filtr medianowy {size}x{size}")

    def apply_sobel_filter(self):
        if self.check_image():
            return
        self.processed_image = self.sobel_filter(self.original_image)
        self.update_preview()
        self.update_histograms()
        self.status_var.set("Zastosowano filtr Sobel")

    def apply_highpass_filter(self):
        if self.check_image():
            return
        self.processed_image = self.high_pass_filter(self.original_image)
        self.update_preview()
        self.update_histograms()
        self.status_var.set("Zastosowano filtr górnoprzepustowy")

    def apply_gaussian_blur(self):
        if self.check_image():
            return
        sigma = self.sigma_var.get()
        self.processed_image = self.gaussian_blur(self.original_image, sigma)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Zastosowano rozmycie Gaussa (σ={sigma})")

    def define_custom_mask(self):
        if self.check_image():
            return

        size = self.custom_size.get()

        # Okno do definiowania maski
        mask_window = tk.Toplevel(self.root)
        mask_window.title(f"Definiowanie Maski {size}x{size}")
        mask_window.geometry("400x500")
        mask_window.transient(self.root)
        mask_window.grab_set()

        entries = []

        for i in range(size):
            row_frame = ttk.Frame(mask_window)
            row_frame.pack(fill=tk.X, padx=10, pady=2)

            row_entries = []
            for j in range(size):
                entry = ttk.Entry(row_frame, width=8)
                entry.insert(0, "1.0" if i == j else "0.0")
                row_entries.append(entry)
                entry.pack(side=tk.LEFT, padx=2)
            entries.append(row_entries)

        def apply_mask():
            try:
                kernel = []
                for i in range(size):
                    row = []
                    for j in range(size):
                        value = float(entries[i][j].get())
                        row.append(value)
                    kernel.append(row)

                kernel = np.array(kernel)
                self.processed_image = self.custom_convolution(
                    self.original_image, kernel
                )
                self.update_preview()
                self.update_histograms()
                self.status_var.set("Zastosowano własną maskę")
                mask_window.destroy()

            except ValueError:
                messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe!")

        ttk.Button(mask_window, text="Zastosuj Maskę", command=apply_mask).pack(pady=10)

    # Metody histogramu
    def apply_histogram_stretching(self):
        if self.check_image():
            return
        self.processed_image = self.histogram_stretching(self.original_image)
        self.update_preview()
        self.update_histograms()
        self.status_var.set("Zastosowano rozszerzenie histogramu")

    def apply_histogram_equalization(self):
        if self.check_image():
            return
        self.processed_image = self.histogram_equalization(self.original_image)
        self.update_preview()
        self.update_histograms()
        self.status_var.set("Zastosowano wyrównanie histogramu")

    # Metody binaryzacji
    def apply_manual_binarization(self):
        if self.check_image():
            return
        threshold = self.manual_threshold.get()
        self.processed_image = self.manual_binarization(self.original_image, threshold)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Zastosowano binaryzację ręczną (próg={threshold})")

    def apply_percent_black(self):
        if self.check_image():
            return
        # Użycie 50% jako domyślnej wartości dla "Percent Black"
        threshold = self.percent_black_selection(self.original_image, percent=0.5)
        self.processed_image = self.manual_binarization(self.original_image, threshold)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Percent Black Selection (próg={threshold})")

    def apply_mean_iterative(self):
        if self.check_image():
            return
        threshold = self.mean_iterative_selection(self.original_image)
        self.processed_image = self.manual_binarization(self.original_image, threshold)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Mean Iterative Selection (próg={threshold})")

    def apply_entropy_selection(self):
        if self.check_image():
            return
        threshold = self.entropy_selection(self.original_image)
        self.processed_image = self.manual_binarization(self.original_image, threshold)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Entropy Selection (próg={threshold})")

    def apply_minimum_error(self):
        if self.check_image():
            return
        threshold = self.minimum_error(self.original_image)
        self.processed_image = self.manual_binarization(self.original_image, threshold)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Minimum Error (próg={threshold})")

    def apply_fuzzy_minimum_error(self):
        if self.check_image():
            return
        threshold = self.fuzzy_minimum_error(self.original_image)
        self.processed_image = self.manual_binarization(self.original_image, threshold)
        self.update_preview()
        self.update_histograms()
        self.status_var.set(f"Fuzzy Minimum Error (próg={threshold})")

    def check_image(self):
        """Sprawdzenie czy obraz jest wczytany"""
        if self.image is None:
            messagebox.showwarning("Ostrzeżenie", "Najpierw wczytaj obraz!")
            return True
        return False

    # Implementacje algorytmów przetwarzania obrazu - FILTRY (z poprzedniego kodu)

    @staticmethod
    def average_filter(image, kernel_size=3):
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            image = image[:, :, np.newaxis]

        pad = kernel_size // 2
        result = np.zeros_like(image, dtype=np.float32)
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

        for c in range(channels):
            padded = np.pad(image[:, :, c], pad, mode="constant", constant_values=0)
            for i in range(height):
                for j in range(width):
                    region = padded[i : i + kernel_size, j : j + kernel_size]
                    result[i, j, c] = np.sum(region * kernel)

        if channels == 1:
            return np.clip(result[:, :, 0], 0, 255).astype(np.uint8)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def median_filter(image, kernel_size=3):
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            image = image[:, :, np.newaxis]

        pad = kernel_size // 2
        result = np.zeros_like(image)

        for c in range(channels):
            padded = np.pad(image[:, :, c], pad, mode="constant", constant_values=0)
            for i in range(height):
                for j in range(width):
                    region = padded[i : i + kernel_size, j : j + kernel_size]
                    result[i, j, c] = np.median(region)

        if channels == 1:
            return result[:, :, 0].astype(np.uint8)
        return result.astype(np.uint8)

    @staticmethod
    def sobel_filter(image):
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        height, width = gray.shape
        pad = 1
        padded = np.pad(gray, pad, mode="constant", constant_values=0)
        gradient = np.zeros_like(gray, dtype=np.float32)

        for i in range(height):
            for j in range(width):
                region = padded[i : i + 3, j : j + 3]
                gx = np.sum(region * sobel_x)
                gy = np.sum(region * sobel_y)
                gradient[i, j] = np.sqrt(gx**2 + gy**2)

        if gradient.max() > 0:
            gradient = (gradient / gradient.max()) * 255
        return gradient.astype(np.uint8)

    @staticmethod
    def high_pass_filter(image):
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            image = image[:, :, np.newaxis]

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        pad = 1
        result = np.zeros_like(image, dtype=np.float32)

        for c in range(channels):
            padded = np.pad(image[:, :, c], pad, mode="constant", constant_values=0)
            for i in range(height):
                for j in range(width):
                    region = padded[i : i + 3, j : j + 3]
                    result[i, j, c] = np.sum(region * kernel)

        if channels == 1:
            return np.clip(result[:, :, 0], 0, 255).astype(np.uint8)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
                / (2 * sigma**2)
            ),
            (size, size),
        )
        return kernel / np.sum(kernel)

    @staticmethod
    def gaussian_blur(image, sigma=1.0, kernel_size=None):
        if kernel_size is None:
            kernel_size = int(6 * sigma) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1

        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            image = image[:, :, np.newaxis]

        pad = kernel_size // 2
        kernel = ImageEnhancementApp.gaussian_kernel(kernel_size, sigma)
        result = np.zeros_like(image, dtype=np.float32)

        for c in range(channels):
            padded = np.pad(image[:, :, c], pad, mode="constant", constant_values=0)
            for i in range(height):
                for j in range(width):
                    region = padded[i : i + kernel_size, j : j + kernel_size]
                    result[i, j, c] = np.sum(region * kernel)

        if channels == 1:
            return np.clip(result[:, :, 0], 0, 255).astype(np.uint8)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def custom_convolution(image, kernel):
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            image = image[:, :, np.newaxis]

        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        result = np.zeros_like(image, dtype=np.float32)

        for c in range(channels):
            padded = np.pad(image[:, :, c], pad, mode="constant", constant_values=0)
            for i in range(height):
                for j in range(width):
                    region = padded[i : i + kernel_size, j : j + kernel_size]
                    result[i, j, c] = np.sum(region * kernel)

        if channels == 1:
            return np.clip(result[:, :, 0], 0, 255).astype(np.uint8)
        return np.clip(result, 0, 255).astype(np.uint8)

    # Implementacje algorytmów - HISTOGRAM
    @staticmethod
    def histogram_stretching(image):
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        min_val = np.min(gray)
        max_val = np.max(gray)

        if max_val == min_val:
            return gray

        stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return stretched

    @staticmethod
    def histogram_equalization(image):
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        # Oblicz histogram
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])

        # Oblicz dystrybuantę (CDF)
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]

        # Użyj mapowania (interpolacji)
        equalized = np.interp(gray.flatten(), bins[:-1], cdf_normalized)
        equalized = equalized.reshape(gray.shape).astype(np.uint8)

        return equalized

    # Implementacje algorytmów - BINARYZACJA
    @staticmethod
    def manual_binarization(image, threshold):
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        binary = np.zeros_like(gray)
        # Piksele powyżej progu stają się białe (255)
        binary[gray > threshold] = 255
        return binary

    @staticmethod
    def percent_black_selection(image, percent=0.5):
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        # Sortuj wartości pikseli, żeby znaleźć kwantyl
        sorted_pixels = np.sort(gray.flatten())

        # Indeks kwantyla (próg)
        index = int(percent * len(sorted_pixels))
        threshold = sorted_pixels[index]

        return threshold

    @staticmethod
    def mean_iterative_selection(image, max_iterations=100, tolerance=1):
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        # Początkowy próg - średnia
        threshold = np.mean(gray)

        for i in range(max_iterations):
            background = gray[gray <= threshold]
            foreground = gray[gray > threshold]

            if len(background) == 0 or len(foreground) == 0:
                break

            mean_background = np.mean(background)
            mean_foreground = np.mean(foreground)

            # Nowy próg to średnia z dwóch średnich
            new_threshold = (mean_background + mean_foreground) / 2

            if abs(new_threshold - threshold) < tolerance:
                break

            threshold = new_threshold

        return int(threshold)

    @staticmethod
    def entropy_selection(image):
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
        hist = hist.astype(float)
        hist /= hist.sum()

        max_entropy = -1
        best_threshold = 0

        for t in range(1, 255):
            # Prawdopodobieństwa klas
            p_background = hist[:t].sum()
            p_foreground = hist[t:].sum()

            if p_background == 0 or p_foreground == 0:
                continue

            # Entropia tła
            entropy_background = 0
            for i in range(t):
                if hist[i] > 0:
                    p = hist[i] / p_background
                    # Entropia (Sum p log p)
                    entropy_background -= p * math.log(p)

            # Entropia obiektu
            entropy_foreground = 0
            for i in range(t, 256):
                if hist[i] > 0:
                    p = hist[i] / p_foreground
                    entropy_foreground -= p * math.log(p)

            total_entropy = entropy_background + entropy_foreground

            if total_entropy > max_entropy:
                max_entropy = total_entropy
                best_threshold = t

        return best_threshold

    @staticmethod
    def minimum_error(image):
        # Implementacja oparta na minimalizacji błędu Gaussa (Kittler i Illingworth)
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
        hist = hist.astype(float)
        hist /= hist.sum()

        min_error = float("inf")
        best_threshold = 0

        for t in range(1, 255):
            p_background = hist[:t].sum()
            p_foreground = hist[t:].sum()

            if p_background == 0 or p_foreground == 0:
                continue

            # Średnie i wariancje
            mean_background = np.sum(np.arange(t) * hist[:t]) / p_background
            mean_foreground = np.sum(np.arange(t, 256) * hist[t:]) / p_foreground

            var_background = (
                np.sum((np.arange(t) - mean_background) ** 2 * hist[:t]) / p_background
            )
            var_foreground = (
                np.sum((np.arange(t, 256) - mean_foreground) ** 2 * hist[t:])
                / p_foreground
            )

            # Wariancje muszą być > 0 dla logarytmu
            if var_background <= 0 or var_foreground <= 0:
                continue

            # Wzór błędu
            error = 1 + 2 * (
                p_background * math.log(var_background)
                + p_foreground * math.log(var_foreground)
            )
            # Uwzględniając prawdopodobieństwa log(P_i)
            error -= 2 * (
                p_background * math.log(p_background)
                + p_foreground * math.log(p_foreground)
            )

            if error < min_error:
                min_error = error
                best_threshold = t

        return best_threshold

    @staticmethod
    def fuzzy_minimum_error(image):
        # Implementacja uproszczona (fuzzy c-means clustering)
        if len(image.shape) == 3:
            gray = ImageEnhancementApp.rgb_to_grayscale(image)
        else:
            gray = image

        hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
        hist = hist.astype(float)
        hist /= hist.sum()

        min_error = float("inf")
        best_threshold = 0

        # Wymaga zewnętrznych narzędzi (Fuzzy C-Means), uproszczona wersja poniżej:
        # Poniższa implementacja jest bardzo uproszczonym przybliżeniem,
        # ponieważ funkcje przynależności powinny być dynamiczne i zależne od t.

        for t in range(1, 255):
            # Uproszczone funkcje przynależności (liniowe, zależne od t)
            # Przykład: stopień przynależności do tła spada liniowo do t
            # stopień przynależności do obiektu rośnie liniowo od t
            background_membership = np.clip(1 - (np.arange(256) - t) / 255, 0, 1)
            foreground_membership = np.clip((np.arange(256) - t) / 255, 0, 1)

            # Normalizacja, by uniknąć p_b=0 lub p_f=0
            power_background = np.sum(hist * background_membership)
            power_foreground = np.sum(hist * foreground_membership)

            if power_background <= 0 or power_foreground <= 0:
                continue

            # Średnie rozmyte
            mean_background = (
                np.sum(hist * background_membership * np.arange(256)) / power_background
            )
            mean_foreground = (
                np.sum(hist * foreground_membership * np.arange(256)) / power_foreground
            )

            # Wariancje rozmyte
            var_background = (
                np.sum(
                    hist
                    * background_membership
                    * (np.arange(256) - mean_background) ** 2
                )
                / power_background
            )
            var_foreground = (
                np.sum(
                    hist
                    * foreground_membership
                    * (np.arange(256) - mean_foreground) ** 2
                )
                / power_foreground
            )

            if var_background <= 0 or var_foreground <= 0:
                continue

            # Wzór błędu (jak dla Minimum Error)
            error = 1 + 2 * (
                power_background * math.log(var_background)
                + power_foreground * math.log(var_foreground)
            )
            error -= 2 * (
                power_background * math.log(power_background)
                + power_foreground * math.log(power_foreground)
            )

            if error < min_error:
                min_error = error
                best_threshold = t

        return best_threshold

    @staticmethod
    def rgb_to_grayscale(image):
        """Konwersja RGB do skali szarości (luminancja)"""
        if len(image.shape) == 3:
            return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return image


def main():
    """Uruchomienie aplikacji"""
    root = tk.Tk()
    app = ImageEnhancementApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
