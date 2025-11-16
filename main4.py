import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
import os


class AplikacjaObrazu:
    def __init__(self, root):
        self.root = root
        self.root.title("Przekształcenia Obrazów - Punktowe i Filtry")
        self.root.geometry("1200x800")

        self.obraz = None
        self.obraz_oryginalny = None
        self.sciezka_pliku = None

        self._konfiguruj_gui()

    def _konfiguruj_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        kontrolny_frame = ttk.LabelFrame(main_frame, text="Kontrola", width=300)
        kontrolny_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        kontrolny_frame.pack_propagate(False)

        preview_frame = ttk.LabelFrame(main_frame, text="Podgląd")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        load_frame = ttk.Frame(kontrolny_frame)
        load_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(load_frame, text="Wczytaj Obraz", command=self.load_image).pack(
            fill=tk.X
        )
        ttk.Button(load_frame, text="Zapisz Obraz", command=self.save_image).pack(
            fill=tk.X, pady=(5, 0)
        )

        ttk.Separator(kontrolny_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=5, pady=10
        )

        notebook = ttk.Notebook(kontrolny_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        point_frame = ttk.Frame(notebook)
        notebook.add(point_frame, text="Przekształcenia Punktowe")
        self._konfiguruj_punktowe(point_frame)

        filter_frame = ttk.Frame(notebook)
        notebook.add(filter_frame, text="Filtry")
        self._konfiguruj_filtry(filter_frame)

        self._konfiguruj_podglad(preview_frame)

        self.status_var = tk.StringVar()
        self.status_var.set("Gotowy")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _konfiguruj_punktowe(self, parent):
        jasnosc_frame = ttk.LabelFrame(parent, text="Zmiana Jasności")
        jasnosc_frame.pack(fill=tk.X, padx=5, pady=5)

        self.zmienna_jasnosc = tk.IntVar(value=0)
        ttk.Scale(
            jasnosc_frame,
            from_=-100,
            to=100,
            variable=self.zmienna_jasnosc,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(
            jasnosc_frame,
            text="Zastosuj Zmianę Jasności",
            command=self.apply_brightness,
        ).pack(fill=tk.X, padx=5, pady=5)

        arytm_frame = ttk.LabelFrame(parent, text="Operacje Arytmetyczne")
        arytm_frame.pack(fill=tk.X, padx=5, pady=5)

        self.wartosc_arytm = tk.DoubleVar(value=10.0)
        ttk.Entry(arytm_frame, textvariable=self.wartosc_arytm).pack(
            fill=tk.X, padx=5, pady=2
        )

        button_frame = ttk.Frame(arytm_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            button_frame, text="Dodaj", command=lambda: self.apply_arithmetic("add")
        ).pack(side=tk.LEFT, expand=True)
        ttk.Button(
            button_frame,
            text="Odejmij",
            command=lambda: self.apply_arithmetic("subtract"),
        ).pack(side=tk.LEFT, expand=True)
        ttk.Button(
            button_frame,
            text="Pomnóż",
            command=lambda: self.apply_arithmetic("multiply"),
        ).pack(side=tk.LEFT, expand=True)
        ttk.Button(
            button_frame,
            text="Podziel",
            command=lambda: self.apply_arithmetic("divide"),
        ).pack(side=tk.LEFT, expand=True)

        szarosc_frame = ttk.LabelFrame(parent, text="Skala Szarości")
        szarosc_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            szarosc_frame, text="Metoda Średniej", command=self.grayscale_average
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            szarosc_frame, text="Metoda Luminancji", command=self.grayscale_luminance
        ).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(parent, text="Przywróć Oryginał", command=self.reset_image).pack(
            fill=tk.X, padx=5, pady=10
        )

    def _konfiguruj_filtry(self, parent):
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

        med_frame = ttk.LabelFrame(parent, text="Filtr Medianowy")
        med_frame.pack(fill=tk.X, padx=5, pady=5)

        self.med_size = tk.IntVar(value=3)
        ttk.Spinbox(
            med_frame, from_=3, to=15, increment=2, textvariable=self.med_size
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            med_frame, text="Zastosuj Filtr Medianowy", command=self.apply_median_filter
        ).pack(fill=tk.X, padx=5, pady=5)

        sobel_frame = ttk.LabelFrame(parent, text="Wykrywanie Krawędzi")
        sobel_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            sobel_frame, text="Filtr Sobel", command=self.apply_sobel_filter
        ).pack(fill=tk.X, padx=5, pady=5)

        highpass_frame = ttk.LabelFrame(parent, text="Wyostrzanie")
        highpass_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            highpass_frame,
            text="Filtr Górnoprzepustowy",
            command=self.apply_highpass_filter,
        ).pack(fill=tk.X, padx=5, pady=5)

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

        custom_frame = ttk.LabelFrame(parent, text="Własna Maska")
        custom_frame.pack(fill=tk.X, padx=5, pady=5)

        self.custom_size = tk.IntVar(value=3)
        ttk.Spinbox(
            custom_frame, from_=3, to=9, increment=2, textvariable=self.custom_size
        ).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(
            custom_frame, text="Zdefiniuj Maskę", command=self.define_custom_mask
        ).pack(fill=tk.X, padx=5, pady=5)

    def _konfiguruj_podglad(self, parent):
        self.preview_frame = ttk.Frame(parent)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, self.preview_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax1.imshow(np.zeros((100, 100, 3)), cmap="gray")
        self.ax1.set_title("Oryginał")
        self.ax1.axis("off")

        self.ax2.imshow(np.zeros((100, 100, 3)), cmap="gray")
        self.ax2.set_title("Wynik")
        self.ax2.axis("off")

        self.canvas.draw()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[
                ("Obrazy", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.ppm *.pgm"),
                ("PPM Files", "*.ppm"),
                ("PGM Files", "*.pgm"),
                ("Wszystkie pliki", "*.*"),
            ],
        )

        if file_path:
            try:
                if file_path.lower().endswith(".ppm"):
                    self.obraz = self.load_ppm(file_path)
                else:
                    pil_image = Image.open(file_path)
                    self.obraz = np.array(pil_image)

                self.obraz_oryginalny = self.obraz.copy()
                self.sciezka_pliku = file_path
                self.update_preview()
                self.status_var.set(
                    f"Wczytano: {os.path.basename(file_path)} - Rozmiar: {self.obraz.shape}"
                )

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać obrazu: {e}")

    def load_ppm(self, filename):
        try:
            with open(filename, "rb") as f:
                header = f.readline().decode("ascii").strip()
                if header != "P6" and header != "P3":
                    raise ValueError("Nieobsługiwany format PPM")

                while True:
                    line = f.readline().decode("ascii")
                    if not line.startswith("#"):
                        break

                size_match = re.match(r"(\d+)\s+(\d+)", line)
                if size_match:
                    width, height = int(size_match.group(1)), int(size_match.group(2))
                else:
                    parts = line.split()
                    width, height = int(parts[0]), int(parts[1])
                    line = f.readline().decode("ascii")

                max_val = int(f.readline().decode("ascii").strip())

                if header == "P6":
                    data = f.read()
                    image = np.frombuffer(data, dtype=np.uint8)
                else:  # P3
                    data = []
                    for line in f:
                        data.extend(map(int, line.decode("ascii").split()))
                    image = np.array(data, dtype=np.uint8)

                if len(image) == width * height * 3:
                    image = image.reshape((height, width, 3))
                else:
                    image = image.reshape(
                        (height, width)
                    )  # Zakładamy PGM jeśli jednowymiarowy

                return image

        except Exception as e:
            raise Exception(f"Błąd PPM: {e}")

    def save_image(self):
        if self.obraz is None:
            messagebox.showwarning("Ostrzeżenie", "Najpierw wczytaj obraz!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Zapisz obraz jako",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("PPM", "*.ppm"),
                ("Wszystkie pliki", "*.*"),
            ],
        )

        if file_path:
            try:
                if file_path.lower().endswith(".ppm"):
                    self.save_ppm(self.obraz, file_path)
                else:
                    pil_image = Image.fromarray(self.obraz.astype("uint8"))
                    pil_image.save(file_path)

                self.status_var.set(f"Zapisano: {os.path.basename(file_path)}")
                messagebox.showinfo("Sukces", "Obraz zapisany pomyślnie!")

            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać obrazu: {e}")

    def save_ppm(self, image_array, filename, format="P6"):
        wysokosc, szerokosc = image_array.shape[0], image_array.shape[1]

        with open(filename, "wb") as f:
            f.write(f"{format}\n".encode("ascii"))
            f.write(f"{szerokosc} {wysokosc}\n".encode("ascii"))
            f.write("255\n".encode("ascii"))

            if format == "P6":
                f.write(image_array.astype(np.uint8).tobytes())
            else:
                for i in range(wysokosc):
                    for j in range(szerokosc):
                        if len(image_array.shape) == 3:
                            r, g, b = image_array[i, j]
                            f.write(f"{r} {g} {b} ".encode("ascii"))
                        else:
                            val = image_array[i, j]
                            f.write(f"{val} {val} {val} ".encode("ascii"))
                    f.write(b"\n")

    def update_preview(self):
        if self.obraz is not None:
            self.ax1.clear()
            self.ax2.clear()

            # Oryginał
            if len(self.obraz_oryginalny.shape) == 2:
                self.ax1.imshow(self.obraz_oryginalny, cmap="gray")
            else:
                self.ax1.imshow(self.obraz_oryginalny)
            self.ax1.set_title("Oryginał")
            self.ax1.axis("off")

            # Wynik
            if len(self.obraz.shape) == 2:
                self.ax2.imshow(self.obraz, cmap="gray")
            else:
                self.ax2.imshow(self.obraz)
            self.ax2.set_title("Wynik")
            self.ax2.axis("off")

            self.canvas.draw()

    # --- Punktowe ---

    def apply_brightness(self):
        if self._sprawdz_obraz():
            return
        wartosc = self.zmienna_jasnosc.get()
        self.obraz = self.add(self.obraz_oryginalny, wartosc)
        self.update_preview()
        self.status_var.set(f"Zastosowano zmianę jasności: {wartosc}")

    def apply_arithmetic(self, operation):
        if self._sprawdz_obraz():
            return
        wartosc = self.wartosc_arytm.get()

        if operation == "add":
            self.obraz = self.add(self.obraz_oryginalny, wartosc)
            op_text = f"Dodawanie: +{wartosc}"
        elif operation == "subtract":
            self.obraz = self.subtract(self.obraz_oryginalny, wartosc)
            op_text = f"Odejmowanie: -{wartosc}"
        elif operation == "multiply":
            self.obraz = self.multiply(self.obraz_oryginalny, wartosc)
            op_text = f"Mnożenie: ×{wartosc}"
        elif operation == "divide":
            if wartosc == 0:
                messagebox.showerror("Błąd", "Nie można dzielić przez zero!")
                return
            self.obraz = self.divide(self.obraz_oryginalny, wartosc)
            op_text = f"Dzielenie: ÷{wartosc}"

        self.update_preview()
        self.status_var.set(f"Zastosowano {op_text}")

    def grayscale_average(self):
        if self._sprawdz_obraz():
            return
        self.obraz = self.grayscale_average_method(self.obraz_oryginalny)
        self.update_preview()
        self.status_var.set("Zastosowano skalę szarości (średnia)")

    def grayscale_luminance(self):
        if self._sprawdz_obraz():
            return
        self.obraz = self.grayscale_luminance_method(self.obraz_oryginalny)
        self.update_preview()
        self.status_var.set("Zastosowano skalę szarości (luminancja)")

    def reset_image(self):
        if self._sprawdz_obraz():
            return
        self.obraz = self.obraz_oryginalny.copy()
        self.update_preview()
        self.status_var.set("Przywrócono oryginalny obraz")

    # --- Filtry ---

    def apply_average_filter(self):
        if self._sprawdz_obraz():
            return
        rozmiar = self.avg_size.get()
        self.obraz = self.average_filter(self.obraz_oryginalny, rozmiar)
        self.update_preview()
        self.status_var.set(f"Zastosowano filtr uśredniający {rozmiar}x{rozmiar}")

    def apply_median_filter(self):
        if self._sprawdz_obraz():
            return
        rozmiar = self.med_size.get()
        self.obraz = self.median_filter(self.obraz_oryginalny, rozmiar)
        self.update_preview()
        self.status_var.set(f"Zastosowano filtr medianowy {rozmiar}x{rozmiar}")

    def apply_sobel_filter(self):
        if self._sprawdz_obraz():
            return
        self.obraz = self.sobel_filter(self.obraz_oryginalny)
        self.update_preview()
        self.status_var.set("Zastosowano filtr Sobel")

    def apply_highpass_filter(self):
        if self._sprawdz_obraz():
            return
        self.obraz = self.high_pass_filter(self.obraz_oryginalny)
        self.update_preview()
        self.status_var.set("Zastosowano filtr górnoprzepustowy")

    def apply_gaussian_blur(self):
        if self._sprawdz_obraz():
            return
        sigma = self.sigma_var.get()
        self.obraz = self.gaussian_blur(self.obraz_oryginalny, sigma)
        self.update_preview()
        self.status_var.set(f"Zastosowano rozmycie Gaussa (σ={sigma})")

    def define_custom_mask(self):
        if self._sprawdz_obraz():
            return

        rozmiar = self.custom_size.get()

        mask_window = tk.Toplevel(self.root)
        mask_window.title(f"Maska {rozmiar}x{rozmiar}")
        mask_window.geometry("300x400")
        mask_window.transient(self.root)
        mask_window.grab_set()

        entries = []

        for i in range(rozmiar):
            row_frame = ttk.Frame(mask_window)
            row_frame.pack(fill=tk.X, padx=10, pady=2)

            row_entries = []
            for j in range(rozmiar):
                entry = ttk.Entry(row_frame, width=5)
                entry.insert(0, "0")
                entry.pack(side=tk.LEFT, padx=2)
                row_entries.append(entry)
            entries.append(row_entries)

        def apply_mask():
            try:
                kernel = []
                for i in range(rozmiar):
                    row = []
                    for j in range(rozmiar):
                        wartosc = float(entries[i][j].get())
                        row.append(wartosc)
                    kernel.append(row)

                kernel = np.array(kernel)
                self.obraz = self.custom_convolution(self.obraz_oryginalny, kernel)
                self.update_preview()
                self.status_var.set("Zastosowano własną maskę")
                mask_window.destroy()

            except ValueError:
                messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe!")

        ttk.Button(mask_window, text="Zastosuj Maskę", command=apply_mask).pack(pady=10)

    def _sprawdz_obraz(self):
        if self.obraz is None:
            messagebox.showwarning("Ostrzeżenie", "Najpierw wczytaj obraz!")
            return True
        return False

    # --- Algorytmy ---

    @staticmethod
    def add(image, value):
        result = image.astype(np.int32) + value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def subtract(image, value):
        result = image.astype(np.int32) - value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def multiply(image, value):
        result = image.astype(np.float32) * value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def divide(image, value):
        result = image.astype(np.float32) / value
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def grayscale_average_method(image):
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
            return gray.astype(np.uint8)
        return image

    @staticmethod
    def grayscale_luminance_method(image):
        # Wzór luminancji
        if len(image.shape) == 3:
            gray = (
                0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            )
            return gray.astype(np.uint8)
        return image

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
        # Najpierw skala szarości, jeśli jest kolorowy
        if len(image.shape) == 3:
            gray = AplikacjaObrazu.grayscale_luminance_method(image)
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
        # Funkcja Gaussa
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
        kernel = AplikacjaObrazu.gaussian_kernel(kernel_size, sigma)
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


def main():
    root = tk.Tk()
    app = AplikacjaObrazu(root)
    root.mainloop()


if __name__ == "__main__":
    main()
