import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import struct
import io

# --- Klasa do obsługi i wczytywania plików PPM (P3 i P6) ---


class PPMImage:
    """Klasa do wczytywania, przechowywania i skalowania danych PPM."""

    def __init__(self):
        self.width = 0
        self.height = 0
        self.max_val = 0
        self.data = None  # Surowe dane pikseli (np. w postaci listy/tablicy)

    def load_ppm(self, filepath):
        """Wczytuje plik PPM P3 lub P6, z obsługą błędów i blokowym czytaniem."""
        try:
            with open(filepath, "rb") as f:
                # 1. Nagłówek (Magic Number)
                magic_number = f.readline().strip().decode("ascii")
                if magic_number not in ("P3", "P6"):
                    raise ValueError(
                        "Nieobsługiwany lub uszkodzony format pliku (nie P3/P6)."
                    )

                # 2. Wymiary i max_val (z pominięciem komentarzy)
                while True:
                    line = f.readline().strip()
                    if line.startswith(b"#"):
                        continue
                    try:
                        dims = line.decode("ascii").split()
                        self.width = int(dims[0])
                        self.height = int(dims[1])
                    except (IndexError, ValueError):
                        raise IOError("Błąd w odczycie wymiarów obrazu.")

                    line = f.readline().strip()
                    if line.startswith(b"#"):
                        continue
                    try:
                        self.max_val = int(line.decode("ascii").split()[0])
                    except (IndexError, ValueError):
                        raise IOError("Błąd w odczycie maksymalnej wartości koloru.")
                    break  # Wymiary i max_val zostały odczytane

                # 3. Odczyt danych pikseli
                pixel_count = self.width * self.height

                if magic_number == "P3":
                    self._read_p3_data(f, pixel_count)
                elif magic_number == "P6":
                    self._read_p6_data(f, pixel_count)

        except FileNotFoundError:
            raise IOError(f"Plik nie znaleziony: {filepath}")
        except Exception as e:
            raise IOError(f"Błąd podczas wczytywania pliku PPM: {e}")

    def _read_p3_data(self, f, pixel_count):
        """Wczytywanie danych PPM P3 (tekstowy, wolniejszy)."""
        # Czytanie reszty pliku "blokowo" jako tekst
        data_str = f.read().decode("ascii", errors="ignore")
        values = data_str.split()

        if len(values) < pixel_count * 3:
            raise EOFError("Plik P3 jest uszkodzony lub niekompletny.")

        self.data = [int(v) for v in values[: pixel_count * 3]]

    def _read_p6_data(self, f, pixel_count):
        """Wczytywanie danych PPM P6 (binarny, wydajniejszy)."""

        # Określenie wielkości bajtów na składową koloru
        bytes_per_component = 1 if self.max_val <= 255 else 2

        # Wielkość oczekiwanych danych w bajtach (blokowe czytanie)
        expected_data_size = pixel_count * 3 * bytes_per_component
        raw_data = f.read(expected_data_size)

        if len(raw_data) < expected_data_size:
            raise EOFError("Plik P6 jest uszkodzony lub niekompletny.")

        # Przetwarzanie danych binarnych
        if bytes_per_component == 1:
            # 8-bitowe składowe (max_val <= 255)
            self.data = list(raw_data)
        else:
            # 16-bitowe składowe (max_val > 255) - Big-Endian (MSB first, jak podano w opisie)
            # Używamy '>' dla Big-Endian (network byte order), 'H' dla unsigned short (2 bajty)
            fmt = f">{pixel_count * 3}H"
            self.data = list(struct.unpack(fmt, raw_data))

        if len(self.data) != pixel_count * 3:
            raise ValueError("Niepoprawna liczba składowych koloru po parsowaniu.")

    def get_scaled_data(self):
        """Skaluje wartości pikseli liniowo do zakresu 0-255."""
        if self.data is None or self.max_val <= 0:
            return None

        scale_factor = 255.0 / self.max_val

        # Liniowe skalowanie
        scaled_data = [int(val * scale_factor) for val in self.data]

        # Konwersja listy do obiektu bytes dla PIL
        return bytes(scaled_data)


# --- Główna aplikacja GUI (Tkinter + Pillow) ---


class ImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Przeglądarka i Edytor Obrazów (PPM/JPEG)")

        self.current_ppm_image = None  # Obiekt PPMImage
        self.current_pil_image = None  # Obiekt PIL Image
        self.display_image = None  # Obiekt ImageTk do wyświetlania
        self.zoom_factor = 3  # Aktualny poziom powiększenia
        self.pan_x = 0  # Przesunięcie w poziomie
        self.pan_y = 0  # Przesunięcie w pionie
        self.pixel_info_label = None  # Etykieta do wyświetlania wartości pikseli

        self._setup_ui()

    def _setup_ui(self):
        # Menu
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Wczytaj PPM/JPEG", command=self.load_image_dialog)
        file_menu.add_command(
            label="Zapisz do JPEG", command=self.save_as_jpeg_dialog, state=tk.DISABLED
        )
        file_menu.add_separator()
        file_menu.add_command(label="Wyjdź", command=self.quit)
        menubar.add_cascade(label="Plik", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Powiększ (+)", command=lambda: self.zoom(2))
        edit_menu.add_command(label="Pomniejsz (-)", command=lambda: self.zoom(1 / 1.2))
        menubar.add_cascade(label="Edycja", menu=edit_menu)

        self.config(menu=menubar)
        self.file_menu = file_menu  # Zachowanie referencji do aktualizacji statusu

        # Obszar roboczy (Canvas)
        self.canvas = tk.Canvas(self, bg="gray", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Etykieta na dole do informacji o pikselu
        self.pixel_info_label = tk.Label(
            self, text="RGB: ", bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.pixel_info_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Bindowanie zdarzeń dla przesuwania (przeciągania myszą) i informacji o pikselach
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<Motion>", self.show_pixel_info)

        # Zmienne do przesuwania
        self._drag_data = {"x": 0, "y": 0}

    def start_pan(self, event):
        """Inicjuje przesuwanie."""
        if self.zoom_factor > 1.0:  # Przesuwanie aktywne tylko przy powiększeniu
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y

    def do_pan(self, event):
        """Wykonuje przesuwanie i odświeża obraz."""
        if self.zoom_factor > 1.0 and self.current_pil_image:
            dx = event.x - self._drag_data["x"]
            dy = event.y - self._drag_data["y"]
            self.pan_x += dx
            self.pan_y += dy
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y
            self.update_display()

    def show_pixel_info(self, event):
        """Wyświetla wartości R, G, B widocznego piksela przy powiększeniu."""
        if self.current_pil_image and self.zoom_factor > 1.0:
            # Współrzędne na canvasie
            canvas_x, canvas_y = event.x, event.y

            # Szerokość/Wysokość wyświetlanego fragmentu
            disp_w = self.canvas.winfo_width() / self.zoom_factor
            disp_h = self.canvas.winfo_height() / self.zoom_factor

            # Obliczenie oryginalnych współrzędnych piksela (uwzględniając pan)
            # pan_x jest w pixach canvasu, a chcemy w pixach oryginalnego obrazu
            img_x = int((canvas_x - self.pan_x) / self.zoom_factor)
            img_y = int((canvas_y - self.pan_y) / self.zoom_factor)

            # Sprawdzenie czy piksel jest w zakresie obrazu
            if (
                0 <= img_x < self.current_pil_image.width
                and 0 <= img_y < self.current_pil_image.height
            ):
                rgb = self.current_pil_image.getpixel((img_x, img_y))
                self.pixel_info_label.config(
                    text=f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]}) @ (Oryg.: {img_x}, {img_y})"
                )
            else:
                self.pixel_info_label.config(text="RGB: Poza obszarem obrazu")
        elif self.current_pil_image:
            self.pixel_info_label.config(
                text="RGB: Powiększ, aby zobaczyć wartości pikseli"
            )
        else:
            self.pixel_info_label.config(text="RGB: ")

    def load_image_dialog(self):
        """Otwiera okno dialogowe do wczytywania plików PPM lub JPEG."""
        filepath = filedialog.askopenfilename(
            defaultextension=".*",
            filetypes=[
                ("Pliki graficzne", "*.ppm *.PPM *.jpg *.jpeg"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not filepath:
            return

        if filepath.lower().endswith((".ppm")):
            self.load_ppm(filepath)
        elif filepath.lower().endswith((".jpg", ".jpeg")):
            self.load_jpeg(filepath)
        else:
            messagebox.showerror(
                "Błąd", "Nieobsługiwany format pliku. Obsługiwane: PPM (P3, P6), JPEG."
            )

    def load_ppm(self, filepath):
        """Wczytuje PPM, używając własnej implementacji."""
        try:
            ppm_img = PPMImage()
            ppm_img.load_ppm(filepath)
            self.current_ppm_image = ppm_img

            # Konwersja do obiektu PIL Image
            scaled_data = ppm_img.get_scaled_data()
            self.current_pil_image = Image.frombytes(
                "RGB", (ppm_img.width, ppm_img.height), scaled_data
            )

            self.zoom_factor = 1.0  # Resetuj powiększenie
            self.pan_x = 0
            self.pan_y = 0
            self.update_display()
            self.file_menu.entryconfig("Zapisz do JPEG", state=tk.NORMAL)
            self.title(f"Przeglądarka i Edytor Obrazów - {filepath}")

        except IOError as e:
            messagebox.showerror("Błąd Wczytywania PPM", str(e))
        except Exception as e:
            messagebox.showerror(
                "Nieoczekiwany Błąd", f"Wystąpił nieoczekiwany błąd: {e}"
            )

    def load_jpeg(self, filepath):
        """Wczytuje JPEG, używając Pillow."""
        try:
            self.current_pil_image = Image.open(filepath).convert("RGB")
            self.current_ppm_image = None  # Resetuj obiekt PPM
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.update_display()
            self.file_menu.entryconfig("Zapisz do JPEG", state=tk.NORMAL)
            self.title(f"Przeglądarka i Edytor Obrazów - {filepath}")

        except Exception as e:
            messagebox.showerror(
                "Błąd Wczytywania JPEG", f"Nie udało się wczytać pliku JPEG: {e}"
            )

    def save_as_jpeg_dialog(self):
        """Otwiera okno dialogowe do zapisu do JPEG z wyborem kompresji."""
        if not self.current_pil_image:
            messagebox.showwarning("Uwaga", "Brak wczytanego obrazu do zapisu.")
            return

        # Wybór stopnia kompresji
        while True:
            compression_quality = simpledialog.askinteger(
                "Kompresja JPEG",
                "Podaj stopień kompresji (1-100, gdzie 100 to najlepsza jakość):",
                parent=self,
                minvalue=1,
                maxvalue=100,
            )
            if compression_quality is None:  # Użytkownik anulował
                return
            if 1 <= compression_quality <= 100:
                break
            messagebox.showwarning("Błąd", "Wartość musi być z zakresu 1 do 100.")

        # Wybór ścieżki zapisu
        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("Pliki JPEG", "*.jpg")]
        )
        if not filepath:
            return

        try:
            self.current_pil_image.save(filepath, "jpeg", quality=compression_quality)
            messagebox.showinfo(
                "Sukces", f"Obraz zapisano jako JPEG z jakością {compression_quality}."
            )
        except Exception as e:
            messagebox.showerror(
                "Błąd Zapisu JPEG", f"Nie udało się zapisać pliku: {e}"
            )

    def zoom(self, factor):
        """Zmienia poziom powiększenia i aktualizuje wyświetlanie."""
        if self.current_pil_image:
            self.zoom_factor *= factor

            # Ograniczenie powiększenia do rozsądnego zakresu (np. 0.1 do 10)
            self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))

            # Reset pan, jeśli wracamy do normalnego widoku
            if self.zoom_factor <= 1.0:
                self.pan_x = 0
                self.pan_y = 0

            self.update_display()

    def update_display(self):
        """Przetwarza obraz dla aktualnego powiększenia i przesuwania i wyświetla go."""
        if not self.current_pil_image:
            return

        img_w, img_h = self.current_pil_image.size
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

        # Obliczenie docelowych wymiarów po powiększeniu
        zoomed_w = int(img_w * self.zoom_factor)
        zoomed_h = int(img_h * self.zoom_factor)

        # Skalowanie obrazu
        resized_img = self.current_pil_image.resize(
            (zoomed_w, zoomed_h),
            Image.NEAREST,  # Użycie NEAREST (sąsiada) dla powiększenia pikselowego
        )

        # Upewnienie się, że przesunięcia są ograniczone (aby obraz nie zniknął)
        max_pan_x = zoomed_w - canvas_w
        max_pan_y = zoomed_h - canvas_h

        # Ograniczenie przesuwania
        if zoomed_w > canvas_w:
            self.pan_x = max(-max_pan_x, min(0, self.pan_x))
        else:
            self.pan_x = int((canvas_w - zoomed_w) / 2)  # Wyśrodkuj

        if zoomed_h > canvas_h:
            self.pan_y = max(-max_pan_y, min(0, self.pan_y))
        else:
            self.pan_y = int((canvas_h - zoomed_h) / 2)  # Wyśrodkuj

        # Wycięcie widocznego fragmentu (Crop - dla przesunięcia i okna)
        if zoomed_w > canvas_w or zoomed_h > canvas_h:
            # Obliczenie współrzędnych wycięcia na powiększonym obrazie
            crop_x1 = -self.pan_x
            crop_y1 = -self.pan_y
            crop_x2 = crop_x1 + canvas_w
            crop_y2 = crop_y1 + canvas_h

            # Ograniczenie współrzędnych do wymiarów powiększonego obrazu
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(zoomed_w, crop_x2)
            crop_y2 = min(zoomed_h, crop_y2)

            cropped_img = resized_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Wyrównanie przesunięcia dla mniejszych obrazów
            if zoomed_w < canvas_w or zoomed_h < canvas_h:
                # W tym przypadku obraz jest mniejszy niż canvas, więc centrowanie już zostało obsłużone
                # Wystarczy wyświetlić
                pass
        else:
            cropped_img = resized_img

        # Konwersja do formatu Tkinter i wyświetlenie
        self.display_image = ImageTk.PhotoImage(cropped_img)
        self.canvas.delete("all")

        # Wyświetlanie obrazu na środku canvasa lub z uwzględnieniem przesunięcia
        # Tylko jeśli obraz jest mniejszy lub równy canvasowi (wyśrodkowanie)
        display_x = 0
        display_y = 0

        if zoomed_w < canvas_w or zoomed_h < canvas_h:
            display_x = self.pan_x
            display_y = self.pan_y

        # W przypadku, gdy obraz był większy, wycinamy i wyświetlamy w (0,0) na canvasie
        self.canvas.create_image(
            display_x, display_y, anchor=tk.NW, image=self.display_image
        )


if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()
