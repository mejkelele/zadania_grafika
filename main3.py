import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from math import floor, ceil

# --- Konwersje Barw ---


def rgb_to_cmyk(R, G, B):
    R_norm, G_norm, B_norm = R / 255.0, G / 255.0, B / 255.0

    if R_norm == 0 and G_norm == 0 and B_norm == 0:
        return 0, 0, 0, 1.0

    K = 1.0 - max(R_norm, G_norm, B_norm)

    if (1.0 - K) == 0:
        return 0, 0, 0, 1.0

    C = (1.0 - R_norm - K) / (1.0 - K)
    M = (1.0 - G_norm - K) / (1.0 - K)
    Y = (1.0 - B_norm - K) / (1.0 - K)

    return (
        max(0, min(1.0, C)),
        max(0, min(1.0, M)),
        max(0, min(1.0, Y)),
        max(0, min(1.0, K)),
    )


def cmyk_to_rgb(C, M, Y, K):
    R_norm = 1.0 - min(1.0, C * (1.0 - K) + K)
    G_norm = 1.0 - min(1.0, M * (1.0 - K) + K)
    B_norm = 1.0 - min(1.0, Y * (1.0 - K) + K)

    R = int(round(R_norm * 255))
    G = int(round(G_norm * 255))
    B = int(round(B_norm * 255))

    return max(0, min(255, R)), max(0, min(255, G)), max(0, min(255, B))


# --- Główna Klasa ---


class AplikacjaKolorow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Konwerter Barw i Kostka RGB 3D")

        # --- Zmienne ---
        self.zmienna_r = tk.IntVar(value=128)
        self.zmienna_g = tk.IntVar(value=128)
        self.zmienna_b = tk.IntVar(value=128)

        self.c_var = tk.DoubleVar()
        self.m_var = tk.DoubleVar()
        self.y_var = tk.DoubleVar()
        self.k_var = tk.DoubleVar()

        self.aktywne_wejscie = "rgb"
        self.aktualny_rgb_int = (128, 128, 128)

        # --- Konfiguracja UI ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._konfiguruj_konwerter(main_frame)
        self._konfiguruj_kostke_rgb(main_frame)

        self._zmieniono_rgb(128)

    # --- Konwerter ---
    def _konfiguruj_konwerter(self, parent):
        ramka_konwertera = ttk.LabelFrame(
            parent, text="Konwerter (RGB ↔ CMYK)", padding="10"
        )
        ramka_konwertera.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        rgb_frame = ttk.LabelFrame(ramka_konwertera, text="RGB (0-255)")
        rgb_frame.pack(pady=5, padx=5, fill=tk.X)
        self.r_slider = self._dodaj_suwak(
            rgb_frame, "R", self.zmienna_r, 255, self._zmieniono_rgb
        )
        self.g_slider = self._dodaj_suwak(
            rgb_frame, "G", self.zmienna_g, 255, self._zmieniono_rgb
        )
        self.b_slider = self._dodaj_suwak(
            rgb_frame, "B", self.zmienna_b, 255, self._zmieniono_rgb
        )

        cmyk_frame = ttk.LabelFrame(ramka_konwertera, text="CMYK (0.0-1.0)")
        cmyk_frame.pack(pady=5, padx=5, fill=tk.X)
        self.c_slider = self._dodaj_suwak(
            cmyk_frame, "C", self.c_var, 1.0, self._zmieniono_cmyk, step=0.01
        )
        self.m_slider = self._dodaj_suwak(
            cmyk_frame, "M", self.m_var, 1.0, self._zmieniono_cmyk, step=0.01
        )
        self.y_slider = self._dodaj_suwak(
            cmyk_frame, "Y", self.y_var, 1.0, self._zmieniono_cmyk, step=0.01
        )
        self.k_slider = self._dodaj_suwak(
            cmyk_frame, "K", self.k_var, 1.0, self._zmieniono_cmyk, step=0.01
        )

        ramka_wynikow = ttk.Frame(ramka_konwertera)
        ramka_wynikow.pack(pady=10, fill=tk.X)

        ttk.Label(
            ramka_wynikow, text="Wybrany Kolor:", font=("Arial", 10, "bold")
        ).pack(pady=5)
        self.color_display = tk.Canvas(
            ramka_wynikow, width=150, height=50, bg="#808080", highlightthickness=1
        )
        self.color_display.pack(pady=5)

        ttk.Label(
            ramka_wynikow, text="Wyniki Konwersji:", font=("Arial", 10, "bold")
        ).pack(pady=5)
        self.rgb_result_label = ttk.Label(ramka_wynikow, text="RGB: ")
        self.rgb_result_label.pack(anchor=tk.W)
        self.cmyk_result_label = ttk.Label(ramka_wynikow, text="CMYK: ")
        self.cmyk_result_label.pack(anchor=tk.W)

    def _dodaj_suwak(self, parent, label_text, var_obj, max_val, callback, step=1):
        wiersz = ttk.Frame(parent)
        wiersz.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(wiersz, text=label_text).pack(side=tk.LEFT, padx=5, anchor=tk.W)

        entry = ttk.Entry(wiersz, textvariable=var_obj, width=7)
        entry.pack(side=tk.RIGHT)

        slider = ttk.Scale(
            wiersz,
            from_=0,
            to=max_val,
            variable=var_obj,
            orient=tk.HORIZONTAL,
            command=callback,
        )
        slider.pack(side=tk.RIGHT, expand=True, fill=tk.X)

        entry.bind("<Return>", lambda event: callback(var_obj.get()))
        entry.bind("<FocusOut>", lambda event: callback(var_obj.get()))

        return slider

    def _zmieniono_rgb(self, value):
        self.aktywne_wejscie = "rgb"
        try:
            R = int(self.zmienna_r.get())
            G = int(self.zmienna_g.get())
            B = int(self.zmienna_b.get())
            self.aktualny_rgb_int = (R, G, B)

            C, M, Y, K = rgb_to_cmyk(R, G, B)

            # Aktualizacja CMYK
            self.c_var.set(round(C, 3))
            self.m_var.set(round(M, 3))
            self.y_var.set(round(Y, 3))
            self.k_var.set(round(K, 3))

            self._odswiez_wyniki(R, G, B, C, M, Y, K)

        except ValueError:
            self.rgb_result_label.config(
                text="Błąd: Wprowadź liczby całkowite (0-255)."
            )

    def _zmieniono_cmyk(self, value):
        self.aktywne_wejscie = "cmyk"
        try:
            C = float(self.c_var.get())
            M = float(self.m_var.get())
            Y = float(self.y_var.get())
            K = float(self.k_var.get())

            R, G, B = cmyk_to_rgb(C, M, Y, K)
            self.aktualny_rgb_int = (R, G, B)

            # Aktualizacja RGB
            self.zmienna_r.set(R)
            self.zmienna_g.set(G)
            self.zmienna_b.set(B)

            self._odswiez_wyniki(R, G, B, C, M, Y, K)

        except ValueError:
            self.cmyk_result_label.config(
                text="Błąd: Wprowadź liczby dziesiętne (0.0-1.0)."
            )

    def _odswiez_wyniki(self, R, G, B, C, M, Y, K):
        kolor_hex = f"#{R:02x}{G:02x}{B:02x}"
        self.color_display.config(bg=kolor_hex)

        self.rgb_result_label.config(text=f"RGB: R={R}, G={G}, B={B}")
        self.cmyk_result_label.config(
            text=f"CMYK: C={C:.3f}, M={M:.3f}, Y={Y:.3f}, K={K:.3f}"
        )

        self.odswiez_kostke_rgb()

    # --- Kostka RGB 3D ---
    def _konfiguruj_kostke_rgb(self, parent):
        ramka_kostki = ttk.LabelFrame(parent, text="Kostka RGB 3D", padding="10")
        ramka_kostki.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Inicjalizacja Matplotlib
        self.figura = plt.Figure(figsize=(6, 6), dpi=100)
        self.osie = self.figura.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figura, master=ramka_kostki)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self._inicjalizuj_osie_kostki()

        # Narzędzia przekroju
        ramka_przekroju = ttk.LabelFrame(ramka_kostki, text="Przekrój", padding="5")
        ramka_przekroju.pack(fill=tk.X, pady=5)

        ttk.Label(ramka_przekroju, text="Wartość (0-255):").pack(side=tk.LEFT, padx=5)
        self.zmienna_przekroju = tk.IntVar(value=128)
        self.suwak_przekroju = ttk.Scale(
            ramka_przekroju,
            from_=0,
            to=255,
            variable=self.zmienna_przekroju,
            orient=tk.HORIZONTAL,
            command=self._rysuj_przekroj,
        )
        self.suwak_przekroju.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.typ_przekroju = tk.StringVar(value="R")
        ttk.Radiobutton(
            ramka_przekroju,
            text="Stałe R",
            variable=self.typ_przekroju,
            value="R",
            command=lambda: self._rysuj_przekroj(self.zmienna_przekroju.get()),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            ramka_przekroju,
            text="Stałe G",
            variable=self.typ_przekroju,
            value="G",
            command=lambda: self._rysuj_przekroj(self.zmienna_przekroju.get()),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            ramka_przekroju,
            text="Stałe B",
            variable=self.typ_przekroju,
            value="B",
            command=lambda: self._rysuj_przekroj(self.zmienna_przekroju.get()),
        ).pack(side=tk.LEFT, padx=5)

    def _inicjalizuj_osie_kostki(self):
        self.osie.set_xlabel("Red")
        self.osie.set_ylabel("Green")
        self.osie.set_zlabel("Blue")
        self.osie.set_xlim(0, 255)
        self.osie.set_ylim(0, 255)
        self.osie.set_zlim(0, 255)
        self.osie.set_title("Kostka RGB")

        resolution = 10
        x = np.linspace(0, 255, resolution)
        y = np.linspace(0, 255, resolution)

        # Rysowanie 6 ścianek
        X, Y = np.meshgrid(x, y)

        # B=0
        Z = np.zeros_like(X)
        self._rysuj_powierzchnie(X, Y, Z, self.osie)

        # B=255
        Z = np.full_like(X, 255)
        self._rysuj_powierzchnie(X, Y, Z, self.osie)

        # G=0
        X_G, Z_G = np.meshgrid(x, y)
        Y_G = np.zeros_like(X_G)
        self._rysuj_powierzchnie(X_G, Y_G, Z_G, self.osie, fixed_comp="G")

        # G=255
        Y_G = np.full_like(X_G, 255)
        self._rysuj_powierzchnie(X_G, Y_G, Z_G, self.osie, fixed_comp="G")

        # R=0
        Y_R, Z_R = np.meshgrid(x, y)
        X_R = np.zeros_like(Y_R)
        self._rysuj_powierzchnie(X_R, Y_R, Z_R, self.osie, fixed_comp="R")

        # R=255
        X_R = np.full_like(Y_R, 255)
        self._rysuj_powierzchnie(X_R, Y_R, Z_R, self.osie, fixed_comp="R")

        self.canvas.draw()

    def _rysuj_powierzchnie(self, X_in, Y_in, Z_in, ax, fixed_comp=None):
        # Mapowanie koloru na podstawie współrzędnych
        if fixed_comp == "G":
            R_map, G_map, B_map = X_in, Y_in, Z_in
        elif fixed_comp == "R":
            R_map, G_map, B_map = X_in, Y_in, Z_in
        else:
            R_map, G_map, B_map = X_in, Y_in, Z_in

        R_norm = R_map / 255.0
        G_norm = G_map / 255.0
        B_norm = B_map / 255.0

        colors = np.stack([R_norm, G_norm, B_norm], axis=-1)

        if fixed_comp == "G":
            ax.plot_surface(
                R_map,
                G_map,
                B_map,
                facecolors=colors,
                shade=False,
                rstride=1,
                cstride=1,
                antialiased=False,
            )
        elif fixed_comp == "R":
            ax.plot_surface(
                R_map,
                G_map,
                B_map,
                facecolors=colors,
                shade=False,
                rstride=1,
                cstride=1,
                antialiased=False,
            )
        else:
            ax.plot_surface(
                X_in,
                Y_in,
                Z_in,
                facecolors=colors,
                shade=False,
                rstride=1,
                cstride=1,
                antialiased=False,
            )

    def odswiez_kostke_rgb(self):
        # Usuń stary punkt i przekrój
        for scatter in self.osie.collections:
            if isinstance(scatter, matplotlib.collections.PathCollection):
                scatter.remove()
        for poly in self.osie.collections:
            if hasattr(poly, "get_label") and poly.get_label() == "section":
                poly.remove()

        R, G, B = self.aktualny_rgb_int

        # Rysuj nowy punkt
        self.osie.scatter(
            R, G, B, color=f"#{R:02x}{G:02x}{B:02x}", s=100, label="Aktualny Kolor"
        )

        # Rysuj przekrój znowu
        if hasattr(self, "_section_plot"):
            self._rysuj_przekroj(self.zmienna_przekroju.get())

        self.canvas.draw()

    def _rysuj_przekroj(self, value):
        for poly in self.osie.collections:
            if hasattr(poly, "get_label") and poly.get_label() == "section":
                poly.remove()

        wartosc = int(self.zmienna_przekroju.get())
        typ = self.typ_przekroju.get()

        resolution = 50
        coord = np.linspace(0, 255, resolution)
        C1, C2 = np.meshgrid(coord, coord)

        if typ == "R":
            X, Y, Z = np.full_like(C1, wartosc), C1, C2
        elif typ == "G":
            X, Y, Z = C1, np.full_like(C1, wartosc), C2
        elif typ == "B":
            X, Y, Z = C1, C2, np.full_like(C1, wartosc)
        else:
            return

        R_map, G_map, B_map = X, Y, Z

        R_norm = R_map / 255.0
        G_norm = G_map / 255.0
        B_norm = B_map / 255.0
        colors = np.stack([R_norm, G_norm, B_norm], axis=-1)

        # Rysowanie przekroju
        self._section_plot = self.osie.plot_surface(
            X,
            Y,
            Z,
            facecolors=colors,
            shade=False,
            rstride=2,
            cstride=2,
            antialiased=False,
            alpha=0.8,  # Przekrój ma być przezroczysty
            label="section",
            zorder=10,
        )
        self.canvas.draw()


if __name__ == "__main__":
    aplikacja = AplikacjaKolorow()
    aplikacja.mainloop()
