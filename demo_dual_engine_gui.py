"""
demo_dual_engine_gui.py - 2D ResNet-Edge Sub-Millisecond MTF Inspector
A universal, open-source optical inspection desktop application.
Allows users to load ANY camera image or edge crop and run instant 2D ResNet-Edge MTF regression.
"""

import os
import sys
import time
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.engine import ResNetEdgeMTFEngine
from training.synthetic_edge_generator import generate_slanted_edge


class ModernMTFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("2D ResNet-Edge MTF Inspector | Sub-Millisecond Neural Optical Regression")
        self.geometry("1560x940")
        self.minsize(1280, 800)
        self.configure(bg="#0f1115")

        # Initialize Neural Engine
        self.engine = ResNetEdgeMTFEngine()

        # State
        self.current_image = None
        self.current_image_gray = None
        self.display_image = None
        self.detected_rois = []
        self.selected_roi_crop = None
        self.current_results = None
        self.target_lpmm = 50.0
        self.pixel_size_mm = 0.00375

        # Drag selection state
        self.drag_start = None
        self.drag_rect_id = None
        self.scale_factor = 1.0

        self._build_ui()
        self._load_default_synthetic_sample()

    def _build_ui(self):
        # 1. Header Bar
        header = tk.Frame(self, bg="#161920", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="⚡ 2D ResNet-Edge MTF Inspector",
            font=("Segoe UI", 14, "bold"),
            fg="#00e5ff",
            bg="#161920"
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle_lbl = tk.Label(
            header,
            text="Sub-Millisecond Edge-to-MTF Deep Convolutional Regression & 4-Stage Physics Decomposition",
            font=("Segoe UI", 10),
            fg="#8c9ba5",
            bg="#161920"
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=5, pady=14)

        # 2. Top Toolbar
        toolbar = tk.Frame(self, bg="#1c212a", height=50)
        toolbar.pack(fill=tk.X, side=tk.TOP, padx=10, pady=6)

        btn_open = tk.Button(
            toolbar,
            text="📁 Open Image",
            font=("Segoe UI", 10, "bold"),
            bg="#0e639c",
            fg="white",
            relief=tk.FLAT,
            padx=14,
            pady=5,
            command=self._on_open_image
        )
        btn_open.pack(side=tk.LEFT, padx=8, pady=6)

        btn_synth = tk.Button(
            toolbar,
            text="🎲 Synthetic Optical Edge",
            font=("Segoe UI", 10),
            bg="#2c313a",
            fg="#d7dae0",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            command=self._on_generate_synthetic
        )
        btn_synth.pack(side=tk.LEFT, padx=5, pady=6)

        btn_detect = tk.Button(
            toolbar,
            text="🔍 Auto-Detect Edges",
            font=("Segoe UI", 10),
            bg="#2c313a",
            fg="#00e5ff",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            command=self._on_auto_detect_edges
        )
        btn_detect.pack(side=tk.LEFT, padx=5, pady=6)

        tk.Label(toolbar, text="Target lp/mm:", font=("Segoe UI", 9), fg="#abb2bf", bg="#1c212a").pack(side=tk.LEFT, padx=(20, 5))
        self.freq_entry = tk.Entry(toolbar, width=6, font=("Segoe UI", 10), bg="#282c34", fg="white", insertbackground="white")
        self.freq_entry.insert(0, "50.0")
        self.freq_entry.pack(side=tk.LEFT, padx=2)

        btn_run = tk.Button(
            toolbar,
            text="🚀 Run ResNet-Edge Inference",
            font=("Segoe UI", 10, "bold"),
            bg="#238636",
            fg="white",
            relief=tk.FLAT,
            padx=16,
            pady=5,
            command=self._on_run_inference
        )
        btn_run.pack(side=tk.LEFT, padx=15, pady=6)

        btn_export = tk.Button(
            toolbar,
            text="💾 Export CSV",
            font=("Segoe UI", 9),
            bg="#2c313a",
            fg="#abb2bf",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            command=self._on_export_csv
        )
        btn_export.pack(side=tk.RIGHT, padx=10, pady=6)

        # 3. Main Workspace Split (Left: Image & ROI Selection, Right: 4-Stage Curves & Metrics)
        main_split = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#0f1115", bd=0, sashwidth=4)
        main_split.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Column: Image Canvas & ROI List
        left_panel = tk.Frame(main_split, bg="#161920", width=520)
        main_split.add(left_panel, minsize=420)

        img_header = tk.Frame(left_panel, bg="#1c212a")
        img_header.pack(fill=tk.X, side=tk.TOP)
        tk.Label(img_header, text="📷 Image View (Click & Drag to Select ROI)", font=("Segoe UI", 10, "bold"), fg="#abb2bf", bg="#1c212a").pack(side=tk.LEFT, padx=10, pady=6)

        self.canvas = tk.Canvas(left_panel, bg="#0a0c10", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # Edge List Frame
        roi_list_frame = tk.Frame(left_panel, bg="#1c212a", height=130)
        roi_list_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 8))
        tk.Label(roi_list_frame, text="Detected ROIs:", font=("Segoe UI", 9, "bold"), fg="#abb2bf", bg="#1c212a").pack(anchor=tk.W, padx=8, pady=3)

        self.roi_listbox = tk.Listbox(roi_list_frame, bg="#0f1115", fg="#00e5ff", height=4, selectbackground="#0e639c", selectforeground="white", font=("Segoe UI", 9), bd=0)
        self.roi_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        self.roi_listbox.bind("<<ListboxSelect>>", self._on_roi_selected)

        # Right Column: 4-Stage Physics Curves & Metrics Dashboard
        right_panel = tk.Frame(main_split, bg="#161920")
        main_split.add(right_panel, minsize=750)

        # Metrics Top Bar
        metrics_bar = tk.Frame(right_panel, bg="#1c212a", height=70)
        metrics_bar.pack(fill=tk.X, side=tk.TOP, padx=10, pady=8)

        self.card_neural_mtf = self._create_metric_card(metrics_bar, "2D ResNet-Edge MTF", "-- %", "#00e5ff")
        self.card_neural_mtf50 = self._create_metric_card(metrics_bar, "Neural MTF50", "-- lp/mm", "#7ee787")
        self.card_classical_mtf = self._create_metric_card(metrics_bar, "Classical ISO MTF", "-- %", "#ffa657")
        self.card_latency = self._create_metric_card(metrics_bar, "Inference Latency", "-- ms", "#d2a8ff")
        self.card_angle = self._create_metric_card(metrics_bar, "Edge Slant Angle", "-- °", "#79c0ff")

        # 4-Stage Physics Matplotlib Canvas
        curves_frame = tk.Frame(right_panel, bg="#161920")
        curves_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor="#161920")
        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10, hspace=0.38, wspace=0.28)

        self.ax_esf = self.fig.add_subplot(2, 2, 1)
        self.ax_lsf = self.fig.add_subplot(2, 2, 2)
        self.ax_fft = self.fig.add_subplot(2, 2, 3)
        self.ax_mtf = self.fig.add_subplot(2, 2, 4)

        for ax, title in zip(
            [self.ax_esf, self.ax_lsf, self.ax_fft, self.ax_mtf],
            ["Stage 1: Normalized ESF", "Stage 2: LSF Bell Curve", "Stage 3: FFT Spectrum", "Stage 4: 2D ResNet-Edge MTF Curve"]
        ):
            ax.set_facecolor("#0d1117")
            ax.set_title(title, fontsize=10, fontweight="bold", color="#d7dae0")
            ax.tick_params(colors="#8b949e", labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=curves_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_metric_card(self, parent, title, value, fg_color):
        frame = tk.Frame(parent, bg="#0d1117", padx=14, pady=6, highlightbackground="#30363d", highlightthickness=1)
        frame.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.BOTH, expand=True)

        t_lbl = tk.Label(frame, text=title, font=("Segoe UI", 8), fg="#8b949e", bg="#0d1117")
        t_lbl.pack(anchor=tk.W)

        v_lbl = tk.Label(frame, text=value, font=("Segoe UI", 13, "bold"), fg=fg_color, bg="#0d1117")
        v_lbl.pack(anchor=tk.W, pady=(2, 0))
        return v_lbl

    def _load_default_synthetic_sample(self):
        """Loads a clean synthetic slanted edge by default so the app displays immediately."""
        roi, freqs, gt_mtf = generate_slanted_edge(width=50, height=50, angle_deg=6.0, blur_sigma=1.2)
        self.current_image = roi
        self.current_image_gray = roi
        self.selected_roi_crop = roi
        self._display_image_on_canvas(roi)
        self._on_run_inference()

    def _display_image_on_canvas(self, img_array):
        if img_array is None:
            return
        if img_array.ndim == 2:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_array.copy()

        # Scale to fit canvas
        canvas_w = max(100, self.canvas.winfo_width() or 480)
        canvas_h = max(100, self.canvas.winfo_height() or 400)
        h, w = img_bgr.shape[:2]

        self.scale_factor = min(canvas_w / w, canvas_h / h, 2.0)
        disp_w = max(1, int(w * self.scale_factor))
        disp_h = max(1, int(h * self.scale_factor))

        resized = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.display_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.display_image, anchor=tk.CENTER)
        self.img_offset_x = (canvas_w - disp_w) // 2
        self.img_offset_y = (canvas_h - disp_h) // 2

    def _on_open_image(self):
        path = filedialog.askopenfilename(
            title="Select Camera Image or Edge Patch",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All Files", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not load image: {path}")
            return
        self.current_image = img
        self.current_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        self.selected_roi_crop = self.current_image_gray
        self.selected_roi_rot = None
        self.detected_rois = []
        self.roi_listbox.delete(0, tk.END)

        self._display_image_on_canvas(img)

        # If it's a small crop, run inference immediately. Otherwise auto-detect.
        h, w = self.current_image_gray.shape
        if w <= 120 and h <= 120:
            self._on_run_inference()
        else:
            self._on_auto_detect_edges()

    def _on_generate_synthetic(self):
        angle = np.random.uniform(4.0, 8.0)
        sigma = np.random.uniform(0.6, 2.5)
        noise = np.random.uniform(0.5, 3.0)
        roi, _, _ = generate_slanted_edge(width=60, height=60, angle_deg=angle, blur_sigma=sigma, noise_sigma=noise)
        self.current_image = roi
        self.current_image_gray = roi
        self.selected_roi_crop = roi
        self.selected_roi_rot = None
        self.detected_rois = []
        self.roi_listbox.delete(0, tk.END)
        self._display_image_on_canvas(roi)
        self._on_run_inference()

    def _on_auto_detect_edges(self):
        if self.current_image_gray is None:
            return
        rois = self.engine.auto_detect_slanted_edges(self.current_image_gray, max_edges=12)
        self.detected_rois = rois
        self.roi_listbox.delete(0, tk.END)

        self._draw_all_detected_boxes()

        for item in rois:
            name, x1, y1, x2, y2, rot, ang = item
            self.roi_listbox.insert(tk.END, f"{name}: ({x1},{y1}) [{x2-x1}x{y2-y1}]")

        if rois:
            self.roi_listbox.selection_set(0)
            self._on_roi_selected(None)

    def _draw_all_detected_boxes(self, selected_idx=0):
        if not hasattr(self, "img_offset_x") or not self.detected_rois:
            return
        self.canvas.delete("roi_box")
        self.canvas.delete("all_roi_boxes")

        for idx, item in enumerate(self.detected_rois):
            name, x1, y1, x2, y2, rot, ang = item
            sx1 = self.img_offset_x + int(x1 * self.scale_factor)
            sy1 = self.img_offset_y + int(y1 * self.scale_factor)
            sx2 = self.img_offset_x + int(x2 * self.scale_factor)
            sy2 = self.img_offset_y + int(y2 * self.scale_factor)

            if idx == selected_idx:
                outline_color = "#00e5ff"
                box_w = 2
                tag_name = "roi_box"
            else:
                outline_color = "#ffa657" if rot == 90 else "#58a6ff"
                box_w = 1
                tag_name = "all_roi_boxes"

            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=outline_color, width=box_w, tags=tag_name)
            self.canvas.create_text(sx1 + 4, max(12, sy1 - 8), text=f"E{idx+1}", fill=outline_color, font=("Segoe UI", 8, "bold"), tags=tag_name, anchor=tk.W)

    def _on_roi_selected(self, event):
        sel = self.roi_listbox.curselection()
        if not sel or not self.detected_rois:
            return
        idx = sel[0]
        item = self.detected_rois[idx]
        name, x1, y1, x2, y2, rot, ang = item

        self.selected_roi_crop = self.current_image_gray[y1:y2, x1:x2]
        self.selected_roi_rot = rot
        self._draw_all_detected_boxes(selected_idx=idx)
        self._on_run_inference(rotate_deg=rot)

    def _draw_roi_rect(self, x1, y1, x2, y2):
        if not hasattr(self, "img_offset_x"):
            return
        sx1 = self.img_offset_x + int(x1 * self.scale_factor)
        sy1 = self.img_offset_y + int(y1 * self.scale_factor)
        sx2 = self.img_offset_x + int(x2 * self.scale_factor)
        sy2 = self.img_offset_y + int(y2 * self.scale_factor)

        self.canvas.delete("roi_box")
        self.canvas.delete("all_roi_boxes")
        self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#00e5ff", width=2, tags="roi_box")

    def _on_canvas_press(self, event):
        self.drag_start = (event.x, event.y)

    def _on_canvas_drag(self, event):
        if not self.drag_start:
            return
        self.canvas.delete("drag_box")
        x0, y0 = self.drag_start
        self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="#7ee787", width=1, dash=(3, 3), tags="drag_box")

    def _on_canvas_release(self, event):
        if not self.drag_start or self.current_image_gray is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None
        self.canvas.delete("drag_box")

        # Map canvas coordinates back to image pixel coordinates
        if not hasattr(self, "img_offset_x"):
            return
        ix1 = int((min(x0, x1) - self.img_offset_x) / self.scale_factor)
        iy1 = int((min(y0, y1) - self.img_offset_y) / self.scale_factor)
        ix2 = int((max(x0, x1) - self.img_offset_x) / self.scale_factor)
        iy2 = int((max(y0, y1) - self.img_offset_y) / self.scale_factor)

        h, w = self.current_image_gray.shape
        ix1, iy1 = max(0, ix1), max(0, iy1)
        ix2, iy2 = min(w, ix2), min(h, iy2)

        if ix2 - ix1 >= 15 and iy2 - iy1 >= 15:
            self.selected_roi_crop = self.current_image_gray[iy1:iy2, ix1:ix2]
            self._draw_roi_rect(ix1, iy1, ix2, iy2)
            self._on_run_inference()

    def _on_run_inference(self, rotate_deg=None):
        crop = self.selected_roi_crop if self.selected_roi_crop is not None else self.current_image_gray
        if crop is None or crop.size < 32:
            return

        try:
            target_f = float(self.freq_entry.get().strip())
        except ValueError:
            target_f = 50.0

        if rotate_deg is None and hasattr(self, "selected_roi_rot"):
            rotate_deg = self.selected_roi_rot

        res = self.engine.analyze_roi_full(
            crop,
            target_freq_lpmm=target_f,
            pixel_size_mm=self.pixel_size_mm,
            rotate_deg=rotate_deg
        )
        self.current_results = res
        self._update_ui_with_results(res)

    def _update_ui_with_results(self, res):
        if res is None:
            return

        neural = res.get("neural")
        classical = res.get("classical")

        # 1. Update Metrics Cards
        if neural:
            self.card_neural_mtf.config(text=f"{neural['NeuralMTFPercent']:.2f} %")
            self.card_neural_mtf50.config(text=f"{neural['NeuralMTF50']:.1f} lp/mm")
            self.card_latency.config(text=f"{neural['InferenceTimeMs']:.3f} ms")
        else:
            self.card_neural_mtf.config(text="-- %")

        if classical and classical.get("FilteredMTF") is not None:
            c_val = classical["FilteredMTF"] * 100.0
            self.card_classical_mtf.config(text=f"{c_val:.2f} %")
            if "EdgeAngleDeg" in classical:
                self.card_angle.config(text=f"{classical['EdgeAngleDeg']:.1f} °")
        else:
            self.card_classical_mtf.config(text="-- %")

        # 2. Update 4-Stage Curves Plot
        for ax in [self.ax_esf, self.ax_lsf, self.ax_fft, self.ax_mtf]:
            ax.cla()
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8b949e", labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

        # Panel 1: ESF
        self.ax_esf.set_title("Stage 1: Normalized ESF", fontsize=9, fontweight="bold", color="#d7dae0")
        if classical and "NormalizedESF" in classical:
            esf = classical["NormalizedESF"]
            self.ax_esf.plot(esf, color="#58a6ff", linewidth=1.8, label="ESF")
            self.ax_esf.set_ylim(-0.05, 1.05)
            self.ax_esf.set_ylabel("Normalized Intensity", color="#8b949e", fontsize=7)

        # Panel 2: LSF (Gaussian bell curve)
        self.ax_lsf.set_title("Stage 2: LSF Bell Curve (Hann Windowed)", fontsize=9, fontweight="bold", color="#d7dae0")
        if classical and "LSFBellCurve" in classical:
            lsf = classical["LSFBellCurve"]
            self.ax_lsf.plot(lsf, color="#7ee787", linewidth=2.0, label="LSF")
            self.ax_lsf.set_ylim(-0.05, max(1.1, np.max(lsf) * 1.1))
            self.ax_lsf.set_ylabel("Line Spread Amplitude", color="#8b949e", fontsize=7)

        # Panel 3: FFT
        self.ax_fft.set_title("Stage 3: FFT Spectrum", fontsize=9, fontweight="bold", color="#d7dae0")
        if classical and "FFTSpectrum" in classical:
            fft_mag = classical["FFTSpectrum"]
            freqs = classical.get("Frequencies", np.linspace(0, 100, len(fft_mag)))
            mask = freqs <= 100.0
            self.ax_fft.plot(freqs[mask], fft_mag[mask], color="#d2a8ff", linewidth=1.8)
            self.ax_fft.set_ylim(-0.05, 1.05)
            self.ax_fft.set_xlabel("Spatial Frequency (lp/mm)", color="#8b949e", fontsize=7)

        # Panel 4: 2D ResNet-Edge Neural MTF vs Classical ISO 12233
        self.ax_mtf.set_title("Stage 4: 2D ResNet-Edge MTF Curve", fontsize=9, fontweight="bold", color="#d7dae0")
        if neural:
            f_n = neural["Frequencies"]
            m_n = neural["NeuralMTFCurve"]
            self.ax_mtf.plot(f_n, m_n * 100.0, color="#00e5ff", linewidth=2.4, label="2D ResNet-Edge (Neural)")

        if classical and "MTFCurve" in classical:
            f_c = classical["Frequencies"]
            m_c = classical["MTFCurve"]
            mask_c = f_c <= 100.0
            self.ax_mtf.plot(f_c[mask_c], m_c[mask_c] * 100.0, color="#ffa657", linestyle="--", linewidth=1.8, label="ISO 12233 Reference")

        target_f = res.get("target_freq_lpmm", 50.0)
        if neural:
            self.ax_mtf.axvline(x=target_f, color="#ff7b72", linestyle=":", alpha=0.7, label=f"Target: {target_f} lp/mm")
            self.ax_mtf.scatter([target_f], [neural["NeuralMTFPercent"]], color="#00e5ff", s=40, zorder=5)

        self.ax_mtf.set_ylim(-2.0, 102.0)
        self.ax_mtf.set_xlim(0, 100)
        self.ax_mtf.set_xlabel("Spatial Frequency (lp/mm)", color="#8b949e", fontsize=7)
        self.ax_mtf.set_ylabel("MTF (%)", color="#8b949e", fontsize=7)
        self.ax_mtf.legend(loc="upper right", fontsize=7, facecolor="#161920", edgecolor="#30363d", labelcolor="#c9d1d9")

        self.plot_canvas.draw_idle()

    def _on_export_csv(self):
        if not self.current_results or not self.current_results.get("neural"):
            messagebox.showwarning("Warning", "No active MTF results to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv")],
            title="Save MTF Curve Data"
        )
        if not path:
            return
        try:
            neural = self.current_results["neural"]
            freqs = neural["Frequencies"]
            curves = neural["NeuralMTFCurve"]
            with open(path, "w") as f:
                f.write("Frequency_lpmm,Neural_MTF,Neural_MTF_Percent\n")
                for fq, val in zip(freqs, curves):
                    f.write(f"{fq:.2f},{val:.6f},{val*100.0:.2f}\n")
            messagebox.showinfo("Success", f"Exported MTF data to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {e}")


def main():
    app = ModernMTFApp()
    app.mainloop()


if __name__ == "__main__":
    main()
