"""
Mic Equalizer - Real-time microphone equalizer for Windows
============================================================

Menangkap audio dari microphone, memproses dengan 10-band graphic
equalizer, lalu mengirim hasilnya ke output device (idealnya virtual
audio cable seperti VB-CABLE) supaya bisa dipakai sebagai "microphone"
oleh aplikasi lain (OBS, Discord, dll).

Requirements (install dulu di Windows):
    pip install sounddevice scipy numpy

Cara jalan (development):
    python mic_equalizer.py

Cara build jadi .exe (di laptop Windows, folder yang sama):
    pip install pyinstaller
    pyinstaller --onefile --windowed --name MicEqualizer mic_equalizer.py

    Hasil .exe ada di folder dist/MicEqualizer.exe

Sebelum pakai:
    1. Install VB-CABLE (gratis): https://vb-audio.com/Cable/
    2. Jalankan MicEqualizer.exe
    3. Pilih Input Device = microphone asli kamu
    4. Pilih Output Device = "CABLE Input (VB-Audio Virtual Cable)"
    5. Di OBS/Discord/aplikasi lain, pilih "CABLE Output (VB-Audio
       Virtual Cable)" sebagai microphone
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import threading

# ----------------------------------------------------------------------
# Konfigurasi band equalizer (10-band graphic EQ, standar ISO)
# ----------------------------------------------------------------------
BAND_FREQS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
Q_FACTOR = 1.4  # lebar band tiap filter peaking

PRESETS = {
    "Flat":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Bass Boost":    [6, 5, 4, 2, 0, 0, 0, 0, 0, 0],
    "Vocal Boost":   [-2, -1, 0, 2, 4, 4, 3, 1, 0, -1],
    "Treble Boost":  [0, 0, 0, 0, 0, 1, 2, 4, 5, 6],
    "Radio Voice":   [-6, -4, -2, 2, 5, 5, 3, 0, -3, -6],
}


def design_peaking_filter(freq, gain_db, q, fs):
    """
    Desain biquad peaking EQ filter (RBJ Audio Cookbook formula).
    Return koefisien (b, a) untuk scipy.signal.lfilter.
    """
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / fs
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


class MicEqualizer:
    def __init__(self):
        self.gains_db = [0.0] * len(BAND_FREQS)
        self.filters = []  # list of (b, a, zi) per band
        self.stream = None
        self.running = False
        self.input_device = None
        self.output_device = None
        self._rebuild_filters()

    def _rebuild_filters(self):
        """Bangun ulang koefisien filter tiap kali gain slider berubah."""
        new_filters = []
        for freq, gain in zip(BAND_FREQS, self.gains_db):
            b, a = design_peaking_filter(freq, gain, Q_FACTOR, SAMPLE_RATE)
            # pertahankan filter state (zi) lama jika ada, biar tidak klik
            zi = signal.lfilter_zi(b, a)
            new_filters.append([b, a, zi])
        self.filters = new_filters

    def set_gain(self, band_index, gain_db):
        self.gains_db[band_index] = gain_db
        b, a = design_peaking_filter(
            BAND_FREQS[band_index], gain_db, Q_FACTOR, SAMPLE_RATE
        )
        # simpan zi lama supaya transisi halus
        old_zi = self.filters[band_index][2] if self.filters else None
        zi = old_zi if old_zi is not None else signal.lfilter_zi(b, a)
        self.filters[band_index] = [b, a, zi]

    def apply_preset(self, name):
        gains = PRESETS[name]
        for i, g in enumerate(gains):
            self.set_gain(i, g)
        return gains

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        mono = indata[:, 0].astype(np.float64)

        processed = mono
        for band in self.filters:
            b, a, zi = band
            processed, band[2] = signal.lfilter(b, a, processed, zi=zi)

        # cegah clipping
        processed = np.clip(processed, -1.0, 1.0)
        outdata[:, 0] = processed.astype(np.float32)
        if outdata.shape[1] > 1:
            for ch in range(1, outdata.shape[1]):
                outdata[:, ch] = outdata[:, 0]

    def start(self, input_device, output_device):
        if self.running:
            return
        self.input_device = input_device
        self.output_device = output_device
        self._rebuild_filters()
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            channels=1,
            device=(input_device, output_device),
            callback=self._audio_callback,
        )
        self.stream.start()
        self.running = True

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False


class EqualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mic Equalizer")
        self.root.geometry("720x480")
        self.root.resizable(False, False)

        self.eq = MicEqualizer()

        self._build_device_selector()
        self._build_sliders()
        self._build_presets()
        self._build_controls()

    # -- device selection --
    def _build_device_selector(self):
        frame = ttk.LabelFrame(self.root, text="Audio Devices")
        frame.pack(fill="x", padx=10, pady=8)

        devices = sd.query_devices()
        input_names = [
            f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0
        ]
        output_names = [
            f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_output_channels"] > 0
        ]

        ttk.Label(frame, text="Input (Mic asli):").grid(row=0, column=0, sticky="w", padx=5, pady=4)
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(frame, textvariable=self.input_var, values=input_names, width=55, state="readonly")
        self.input_combo.grid(row=0, column=1, padx=5, pady=4)
        if input_names:
            self.input_combo.current(0)

        ttk.Label(frame, text="Output (Virtual Cable):").grid(row=1, column=0, sticky="w", padx=5, pady=4)
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(frame, textvariable=self.output_var, values=output_names, width=55, state="readonly")
        self.output_combo.grid(row=1, column=1, padx=5, pady=4)
        # coba auto-pilih device yang namanya mengandung "CABLE"
        for idx, name in enumerate(output_names):
            if "cable" in name.lower():
                self.output_combo.current(idx)
                break
        else:
            if output_names:
                self.output_combo.current(0)

    # -- band sliders --
    def _build_sliders(self):
        frame = ttk.LabelFrame(self.root, text="10-Band Equalizer (dB)")
        frame.pack(fill="both", expand=True, padx=10, pady=8)

        self.sliders = []
        for i, freq in enumerate(BAND_FREQS):
            col = ttk.Frame(frame)
            col.grid(row=0, column=i, padx=6, pady=6)

            label = "{}Hz".format(freq) if freq < 1000 else "{}kHz".format(freq // 1000)
            ttk.Label(col, text=label).pack()

            slider = ttk.Scale(col, from_=12, to=-12, orient="vertical", length=220)
            slider.pack()
            self.sliders.append(slider)

            val_label = ttk.Label(col, text="0 dB")
            val_label.pack()
            slider.value_label = val_label

            # set nilai awal dulu, BARU pasang command callback,
            # supaya slider.set(0) tidak memicu _on_slider_change
            # sebelum self.sliders & value_label selesai disiapkan
            slider.set(0)
            slider.config(command=lambda val, idx=i: self._on_slider_change(idx, val))

    def _on_slider_change(self, band_index, value):
        gain = float(value)
        self.eq.set_gain(band_index, gain)
        self.sliders[band_index].value_label.config(text=f"{gain:.1f} dB")

    # -- presets --
    def _build_presets(self):
        frame = ttk.LabelFrame(self.root, text="Presets")
        frame.pack(fill="x", padx=10, pady=4)
        for name in PRESETS:
            ttk.Button(frame, text=name, command=lambda n=name: self._apply_preset(n)).pack(side="left", padx=4, pady=4)

    def _apply_preset(self, name):
        gains = self.eq.apply_preset(name)
        for i, g in enumerate(gains):
            self.sliders[i].set(g)
            self.sliders[i].value_label.config(text=f"{g:.1f} dB")

    # -- start/stop --
    def _build_controls(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=10, pady=10)

        self.status_label = ttk.Label(frame, text="Status: berhenti", foreground="red")
        self.status_label.pack(side="left", padx=5)

        self.toggle_btn = ttk.Button(frame, text="Start", command=self._toggle)
        self.toggle_btn.pack(side="right", padx=5)

    def _toggle(self):
        if self.eq.running:
            self.eq.stop()
            self.toggle_btn.config(text="Start")
            self.status_label.config(text="Status: berhenti", foreground="red")
        else:
            try:
                in_idx = int(self.input_var.get().split(":")[0])
                out_idx = int(self.output_var.get().split(":")[0])
            except (ValueError, IndexError):
                messagebox.showerror("Error", "Pilih input dan output device dulu.")
                return
            try:
                self.eq.start(in_idx, out_idx)
            except Exception as e:
                messagebox.showerror("Gagal Start", str(e))
                return
            self.toggle_btn.config(text="Stop")
            self.status_label.config(text="Status: berjalan", foreground="green")

    def on_close(self):
        if self.eq.running:
            self.eq.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = EqualizerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
