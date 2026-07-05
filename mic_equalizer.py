"""
Mic Equalizer PRO - Real-time microphone equalizer for Windows
================================================================

Fitur:
    - 10-band graphic equalizer
    - Echo (delay) effect dengan kontrol Mix, Delay Time, Feedback
    - Reverb effect (Schroeder-style: comb + allpass filters) dengan
      kontrol Mix dan Room Size
    - VU meter LED real-time
    - Tampilan gelap ala hardware equalizer

Requirements (install dulu di Windows):
    pip install sounddevice scipy numpy

Cara jalan (development):
    python mic_equalizer.py

Cara build jadi .exe:
    pip install pyinstaller
    pyinstaller --onedir --windowed --name MicEqualizer mic_equalizer.py

Sebelum pakai:
    1. Install VB-CABLE (gratis): https://vb-audio.com/Cable/
    2. Jalankan MicEqualizer.exe
    3. Pilih Input Device = microphone asli kamu
    4. Pilih Output Device = "CABLE Input (VB-Audio Virtual Cable)"
       (atau "Speakers/Headphones" laptop kalau mau keluar lewat kabel
       fisik ke line-in HP)
    5. Di OBS/Discord/TikTok, pilih "CABLE Output" sebagai microphone
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.signal as signal
import sounddevice as sd

COLOR_BG = "#1a1a1e"
COLOR_PANEL = "#232329"
COLOR_PANEL_BORDER = "#333338"
COLOR_ACCENT = "#00e676"
COLOR_ACCENT_DIM = "#0a5c33"
COLOR_TEXT = "#e8e8ec"
COLOR_TEXT_DIM = "#8a8a92"
COLOR_SLIDER_TROUGH = "#0d0d10"
COLOR_LED_GREEN = "#00e676"
COLOR_LED_YELLOW = "#ffd600"
COLOR_LED_RED = "#ff3d3d"
COLOR_LED_OFF = "#2a2a30"

BAND_FREQS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
Q_FACTOR = 1.4

PRESETS = {
    "Flat":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Bass Boost":    [6, 5, 4, 2, 0, 0, 0, 0, 0, 0],
    "Vocal Boost":   [-2, -1, 0, 2, 4, 4, 3, 1, 0, -1],
    "Treble Boost":  [0, 0, 0, 0, 0, 1, 2, 4, 5, 6],
    "Radio Voice":   [-6, -4, -2, 2, 5, 5, 3, 0, -3, -6],
}


def design_peaking_filter(freq, gain_db, q, fs):
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


class EchoEffect:
    def __init__(self, fs, max_delay_sec=1.5):
        self.fs = fs
        self.buffer = np.zeros(int(fs * max_delay_sec), dtype=np.float64)
        self.write_idx = 0
        self.delay_ms = 300.0
        self.feedback = 0.35
        self.mix = 0.0

    def set_params(self, delay_ms=None, feedback=None, mix=None):
        if delay_ms is not None:
            self.delay_ms = delay_ms
        if feedback is not None:
            self.feedback = feedback
        if mix is not None:
            self.mix = mix

    def process(self, block):
        n = len(block)
        buf_len = len(self.buffer)
        delay_samples = int(self.delay_ms / 1000.0 * self.fs)
        delay_samples = max(1, min(delay_samples, buf_len - 1))
        out = np.empty(n, dtype=np.float64)

        for i in range(n):
            read_idx = (self.write_idx - delay_samples) % buf_len
            delayed = self.buffer[read_idx]
            wet = block[i] + delayed * self.feedback
            self.buffer[self.write_idx] = wet
            out[i] = block[i] * (1 - self.mix) + delayed * self.mix
            self.write_idx = (self.write_idx + 1) % buf_len

        return out


class CombFilter:
    def __init__(self, delay_samples, feedback):
        self.buffer = np.zeros(delay_samples, dtype=np.float64)
        self.idx = 0
        self.feedback = feedback

    def process_sample(self, x):
        y = self.buffer[self.idx]
        self.buffer[self.idx] = x + y * self.feedback
        self.idx = (self.idx + 1) % len(self.buffer)
        return y


class AllpassFilter:
    def __init__(self, delay_samples, feedback=0.5):
        self.buffer = np.zeros(delay_samples, dtype=np.float64)
        self.idx = 0
        self.feedback = feedback

    def process_sample(self, x):
        buffered = self.buffer[self.idx]
        y = -self.feedback * x + buffered
        self.buffer[self.idx] = x + buffered * self.feedback
        self.idx = (self.idx + 1) % len(self.buffer)
        return y


class ReverbEffect:
    COMB_DELAYS_MS = [37.0, 41.1, 43.7, 45.0]
    ALLPASS_DELAYS_MS = [5.0, 1.7]

    def __init__(self, fs):
        self.fs = fs
        self.room_size = 0.5
        self.mix = 0.0
        self._build_filters()

    def _build_filters(self):
        fb = 0.7 + self.room_size * 0.28
        self.combs = [
            CombFilter(int(ms / 1000 * self.fs), fb) for ms in self.COMB_DELAYS_MS
        ]
        self.allpasses = [
            AllpassFilter(int(ms / 1000 * self.fs), 0.5) for ms in self.ALLPASS_DELAYS_MS
        ]

    def set_params(self, room_size=None, mix=None):
        rebuild = False
        if room_size is not None and room_size != self.room_size:
            self.room_size = room_size
            rebuild = True
        if mix is not None:
            self.mix = mix
        if rebuild:
            self._build_filters()

    def process(self, block):
        n = len(block)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            x = block[i]
            wet = sum(c.process_sample(x) for c in self.combs) / len(self.combs)
            for ap in self.allpasses:
                wet = ap.process_sample(wet)
            out[i] = x * (1 - self.mix) + wet * self.mix
        return out


class MicEqualizer:
    def __init__(self):
        self.gains_db = [0.0] * len(BAND_FREQS)
        self.filters = []
        self.stream = None
        self.running = False
        self.echo = EchoEffect(SAMPLE_RATE)
        self.reverb = ReverbEffect(SAMPLE_RATE)
        self.last_level = 0.0
        self._rebuild_filters()

    def _rebuild_filters(self):
        new_filters = []
        for freq, gain in zip(BAND_FREQS, self.gains_db):
            b, a = design_peaking_filter(freq, gain, Q_FACTOR, SAMPLE_RATE)
            zi = signal.lfilter_zi(b, a)
            new_filters.append([b, a, zi])
        self.filters = new_filters

    def set_gain(self, band_index, gain_db):
        self.gains_db[band_index] = gain_db
        b, a = design_peaking_filter(BAND_FREQS[band_index], gain_db, Q_FACTOR, SAMPLE_RATE)
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

        if self.echo.mix > 0:
            processed = self.echo.process(processed)

        if self.reverb.mix > 0:
            processed = self.reverb.process(processed)

        processed = np.clip(processed, -1.0, 1.0)

        rms = float(np.sqrt(np.mean(processed ** 2)) + 1e-9)
        self.last_level = rms

        outdata[:, 0] = processed.astype(np.float32)
        if outdata.shape[1] > 1:
            for ch in range(1, outdata.shape[1]):
                outdata[:, ch] = outdata[:, 0]

    def start(self, input_device, output_device):
        if self.running:
            return
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
        self.last_level = 0.0


class EqualizerGUI:
    NUM_LEDS = 14

    def __init__(self, root):
        self.root = root
        self.root.title("MIC EQUALIZER")
        self.root.geometry("900x680")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        self._setup_style()
        self.eq = MicEqualizer()

        self._build_header()
        self._build_device_selector()
        self._build_vu_meter()
        self._build_sliders()
        self._build_effects()
        self._build_presets()
        self._build_controls()

        self._poll_vu_meter()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=COLOR_PANEL)
        style.configure("Dark.TLabelframe", background=COLOR_PANEL, bordercolor=COLOR_PANEL_BORDER)
        style.configure("Dark.TLabelframe.Label", background=COLOR_PANEL, foreground=COLOR_ACCENT,
                         font=("Consolas", 10, "bold"))
        style.configure("Dark.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Consolas", 9))
        style.configure("Dark.TCombobox", fieldbackground=COLOR_SLIDER_TROUGH, background=COLOR_PANEL,
                         foreground=COLOR_TEXT)
        style.configure("Accent.TButton", background=COLOR_ACCENT_DIM, foreground=COLOR_TEXT,
                         font=("Consolas", 9, "bold"))
        style.map("Accent.TButton", background=[("active", COLOR_ACCENT)])

    def _build_header(self):
        frame = tk.Frame(self.root, bg=COLOR_BG)
        frame.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(frame, text="MIC EQUALIZER", bg=COLOR_BG, fg=COLOR_ACCENT,
                 font=("Consolas", 20, "bold")).pack(side="left")
        tk.Label(frame, text="  //  live streaming edition", bg=COLOR_BG, fg=COLOR_TEXT_DIM,
                 font=("Consolas", 11)).pack(side="left", pady=(6, 0))

    def _build_device_selector(self):
        frame = ttk.LabelFrame(self.root, text="AUDIO DEVICES", style="Dark.TLabelframe")
        frame.pack(fill="x", padx=14, pady=6)

        devices = sd.query_devices()
        input_names = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        output_names = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_output_channels"] > 0]

        ttk.Label(frame, text="Input (Mic asli):", style="Dark.TLabel").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(frame, textvariable=self.input_var, values=input_names,
                                         width=55, state="readonly", style="Dark.TCombobox")
        self.input_combo.grid(row=0, column=1, padx=8, pady=6)
        if input_names:
            self.input_combo.current(0)

        ttk.Label(frame, text="Output (Virtual Cable):", style="Dark.TLabel").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(frame, textvariable=self.output_var, values=output_names,
                                          width=55, state="readonly", style="Dark.TCombobox")
        self.output_combo.grid(row=1, column=1, padx=8, pady=6)
        for idx, name in enumerate(output_names):
            if "cable" in name.lower():
                self.output_combo.current(idx)
                break
        else:
            if output_names:
                self.output_combo.current(0)

    def _build_vu_meter(self):
        frame = ttk.LabelFrame(self.root, text="INPUT LEVEL", style="Dark.TLabelframe")
        frame.pack(fill="x", padx=14, pady=6)

        self.vu_canvas = tk.Canvas(frame, width=860, height=34, bg=COLOR_PANEL, highlightthickness=0)
        self.vu_canvas.pack(padx=8, pady=8)

        self.led_ids = []
        led_w = 56
        gap = 6
        for i in range(self.NUM_LEDS):
            x0 = 4 + i * (led_w + gap)
            led = self.vu_canvas.create_rectangle(x0, 4, x0 + led_w, 30, fill=COLOR_LED_OFF, outline="")
            self.led_ids.append(led)

    def _poll_vu_meter(self):
        level = self.eq.last_level if self.eq.running else 0.0
        db = 20 * np.log10(max(level, 1e-6))
        ratio = np.clip((db + 50) / 50, 0.0, 1.0)
        lit = int(ratio * self.NUM_LEDS)

        for i, led in enumerate(self.led_ids):
            if i < lit:
                if i < self.NUM_LEDS - 4:
                    color = COLOR_LED_GREEN
                elif i < self.NUM_LEDS - 2:
                    color = COLOR_LED_YELLOW
                else:
                    color = COLOR_LED_RED
            else:
                color = COLOR_LED_OFF
            self.vu_canvas.itemconfig(led, fill=color)

        self.root.after(60, self._poll_vu_meter)

    def _build_sliders(self):
        frame = ttk.LabelFrame(self.root, text="10-BAND EQUALIZER (dB)", style="Dark.TLabelframe")
        frame.pack(fill="both", expand=True, padx=14, pady=6)

        inner = tk.Frame(frame, bg=COLOR_PANEL)
        inner.pack(padx=8, pady=6)

        self.sliders = []
        for i, freq in enumerate(BAND_FREQS):
            col = tk.Frame(inner, bg=COLOR_PANEL)
            col.grid(row=0, column=i, padx=8)

            label = "{}Hz".format(freq) if freq < 1000 else "{}kHz".format(freq // 1000)
            tk.Label(col, text=label, bg=COLOR_PANEL, fg=COLOR_ACCENT,
                     font=("Consolas", 9, "bold")).pack()

            slider = tk.Scale(
                col, from_=12, to=-12, orient="vertical", length=200,
                resolution=0.5, bg=COLOR_ACCENT, fg=COLOR_TEXT,
                troughcolor=COLOR_SLIDER_TROUGH, highlightthickness=0,
                activebackground="#5dffb0", showvalue=False,
                sliderrelief="raised", sliderlength=26, bd=2,
            )
            slider.set(0)
            slider.pack()
            self.sliders.append(slider)

            val_label = tk.Label(col, text="0.0 dB", bg=COLOR_PANEL, fg=COLOR_TEXT_DIM, font=("Consolas", 8))
            val_label.pack(pady=(2, 0))
            slider.value_label = val_label

            slider.config(command=lambda val, idx=i: self._on_slider_change(idx, val))

    def _on_slider_change(self, band_index, value):
        gain = float(value)
        self.eq.set_gain(band_index, gain)
        self.sliders[band_index].value_label.config(text=f"{gain:.1f} dB")

    def _build_effects(self):
        frame = tk.Frame(self.root, bg=COLOR_BG)
        frame.pack(fill="x", padx=14, pady=6)

        echo_frame = ttk.LabelFrame(frame, text="ECHO", style="Dark.TLabelframe")
        echo_frame.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.echo_mix = self._build_effect_slider(echo_frame, "Mix", 0, 0, 100, self._on_echo_change)
        self.echo_delay = self._build_effect_slider(echo_frame, "Delay (ms)", 1, 50, 1000, self._on_echo_change, default=300)
        self.echo_feedback = self._build_effect_slider(echo_frame, "Feedback", 2, 0, 90, self._on_echo_change, default=35)

        reverb_frame = ttk.LabelFrame(frame, text="REVERB", style="Dark.TLabelframe")
        reverb_frame.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.reverb_mix = self._build_effect_slider(reverb_frame, "Mix", 0, 0, 100, self._on_reverb_change)
        self.reverb_room = self._build_effect_slider(reverb_frame, "Room Size", 1, 0, 100, self._on_reverb_change, default=50)

    def _build_effect_slider(self, parent, label, row, from_, to, callback, default=0):
        tk.Label(parent, text=label, bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Consolas", 9)).grid(
            row=row, column=0, sticky="w", padx=8, pady=4)
        slider = tk.Scale(
            parent, from_=from_, to=to, orient="horizontal", length=300,
            bg=COLOR_ACCENT, fg=COLOR_TEXT, troughcolor=COLOR_SLIDER_TROUGH,
            highlightthickness=0, activebackground="#5dffb0", showvalue=True,
            sliderrelief="raised", sliderlength=26, bd=2, command=callback,
        )
        slider.set(default)
        slider.grid(row=row, column=1, padx=8, pady=4)
        return slider

    def _on_echo_change(self, _=None):
        self.eq.echo.set_params(
            delay_ms=self.echo_delay.get(),
            feedback=self.echo_feedback.get() / 100.0,
            mix=self.echo_mix.get() / 100.0,
        )

    def _on_reverb_change(self, _=None):
        self.eq.reverb.set_params(
            room_size=self.reverb_room.get() / 100.0,
            mix=self.reverb_mix.get() / 100.0,
        )

    def _build_presets(self):
        frame = ttk.LabelFrame(self.root, text="PRESETS", style="Dark.TLabelframe")
        frame.pack(fill="x", padx=14, pady=6)
        for name in PRESETS:
            ttk.Button(frame, text=name, style="Accent.TButton",
                       command=lambda n=name: self._apply_preset(n)).pack(side="left", padx=6, pady=8)

    def _apply_preset(self, name):
        gains = self.eq.apply_preset(name)
        for i, g in enumerate(gains):
            self.sliders[i].set(g)
            self.sliders[i].value_label.config(text=f"{g:.1f} dB")

    def _build_controls(self):
        frame = tk.Frame(self.root, bg=COLOR_BG)
        frame.pack(fill="x", padx=14, pady=12)

        self.status_dot = tk.Canvas(frame, width=16, height=16, bg=COLOR_BG, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 6))
        self.status_dot_id = self.status_dot.create_oval(2, 2, 14, 14, fill=COLOR_LED_RED, outline="")

        self.status_label = tk.Label(frame, text="STOPPED", bg=COLOR_BG, fg=COLOR_TEXT_DIM,
                                      font=("Consolas", 10, "bold"))
        self.status_label.pack(side="left")

        self.toggle_btn = ttk.Button(frame, text="START", style="Accent.TButton", command=self._toggle)
        self.toggle_btn.pack(side="right")

    def _toggle(self):
        if self.eq.running:
            self.eq.stop()
            self.toggle_btn.config(text="START")
            self.status_label.config(text="STOPPED", fg=COLOR_TEXT_DIM)
            self.status_dot.itemconfig(self.status_dot_id, fill=COLOR_LED_RED)
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
            self.toggle_btn.config(text="STOP")
            self.status_label.config(text="RUNNING", fg=COLOR_ACCENT)
            self.status_dot.itemconfig(self.status_dot_id, fill=COLOR_LED_GREEN)

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
