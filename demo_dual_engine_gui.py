"""
demo_dual_engine_gui.py - Modern Industrial GUI for Dual-Engine MTF
Features:
1. Visual Comparison: Static (Red) vs AI Dynamic 50/50 Luminance-Balanced ROI (Cyan)
2. Detailed 4-Stage Physics Pipeline Breakdown:
   - Panel 1: Normalized ESF (Edge Spread Function, 0.0 to 1.0)
   - Panel 2: LSF (Line Spread Function with Hann/Hamming window)
   - Panel 3: FFT Magnitude Spectrum
   - Panel 4: Normalized MTF Curve (Strictly bounded in 0% to 100%)
3. Neural PINN vs Classical ISO 12233 Real-Time Dual-Engine Consensus
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dynamic_roi_engine import process_full_image_static_vs_dynamic
from dual_overlay_visualizer import generate_dual_overlay_image
from inference.dual_engine import DualEngineMTF


class DualEngineApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Deep Keypoint + PINN-MTF Dual Engine Inspector")
        self.geometry("1560x940")
        self.configure(bg="#121417")

        self.neural_engine = DualEngineMTF()
        self.current_image_path = None
        self.current_image_gray = None
        self.current_results = None
        self.overlay_mode = "both"  # 'both', 'static', 'dynamic'

        self._build_ui()

    def _build_ui(self):
        # 1. Header Bar
        header = tk.Frame(self, bg="#1a1d21", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="🎯 DEEP KEYPOINT + PINN-MTF DUAL-ENGINE INSPECTOR",
            font=("Segoe UI", 14, "bold"),
            fg="#00ffff",
            bg="#1a1d21"
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle_lbl = tk.Label(
            header,
            text="Subpixel 50/50 Luminance Equalization & 4-Stage Physics Decomposition (ESF → LSF → FFT → MTF)",
            font=("Segoe UI", 10),
            fg="#8a99a8",
            bg="#1a1d21"
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=5, pady=14)

        # 2. Control Toolbar
        toolbar = tk.Frame(self, bg="#21252b", height=50)
        toolbar.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5)

        btn_open = tk.Button(
            toolbar,
            text="📁 Select Image",
            font=("Segoe UI", 10, "bold"),
            bg="#0e639c",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5,
            command=self._on_select_image
        )
        btn_open.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_run = tk.Button(
            toolbar,
            text="⚡ Run Dynamic 50/50 Dual-Engine Analysis",
            font=("Segoe UI", 10, "bold"),
            bg="#2da44e",
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5,
            state=tk.DISABLED,
            command=self._on_run_analysis
        )
        self.btn_run.pack(side=tk.LEFT, padx=10, pady=8)

        # Overlay Mode Radio Buttons
        mode_frame = tk.Frame(toolbar, bg="#21252b")
        mode_frame.pack(side=tk.LEFT, padx=25, pady=8)

        tk.Label(mode_frame, text="Overlay Mode:", font=("Segoe UI", 9, "bold"), fg="#8a99a8", bg="#21252b").pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value="both")
        r_both = tk.Radiobutton(mode_frame, text="Dual (Static + Dynamic)", variable=self.mode_var, value="both", fg="#e6edf3", bg="#21252b", selectcolor="#0d1117", activebackground="#21252b", command=self._on_change_overlay_mode)
        r_both.pack(side=tk.LEFT, padx=5)

        r_dyn = tk.Radiobutton(mode_frame, text="AI Dynamic 50/50 Only", variable=self.mode_var, value="dynamic", fg="#00ffff", bg="#21252b", selectcolor="#0d1117", activebackground="#21252b", command=self._on_change_overlay_mode)
        r_dyn.pack(side=tk.LEFT, padx=5)

        r_stat = tk.Radiobutton(mode_frame, text="Static Prior Only", variable=self.mode_var, value="static", fg="#ff6b6b", bg="#21252b", selectcolor="#0d1117", activebackground="#21252b", command=self._on_change_overlay_mode)
        r_stat.pack(side=tk.LEFT, padx=5)

        self.status_badge = tk.Label(
            toolbar,
            text="READY",
            font=("Segoe UI", 11, "bold"),
            bg="#30363d",
            fg="#e6edf3",
            padx=12,
            pady=4
        )
        self.status_badge.pack(side=tk.RIGHT, padx=15, pady=8)

        # 3. Main Split View
        main_paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#121417", sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Column: Image Overlay Display
        left_frame = tk.Frame(main_paned, bg="#181a1f")
        main_paned.add(left_frame, minsize=520)

        img_hdr = tk.Label(
            left_frame,
            text="Visual Comparison: Static (Red) vs AI Dynamic 50/50 (Cyan)",
            font=("Segoe UI", 11, "bold"),
            fg="#e6edf3",
            bg="#181a1f"
        )
        img_hdr.pack(anchor="w", padx=10, pady=8)

        self.canvas_img = tk.Canvas(left_frame, bg="#0d1117", highlightthickness=0)
        self.canvas_img.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Right Column: 4-Panel Physics Pipeline + Detailed Metrics Table
        right_frame = tk.Frame(main_paned, bg="#181a1f")
        main_paned.add(right_frame, minsize=780)

        # 4-Panel Physics Figure: (2x2 Grid)
        # Panel 1: Normalized ESF | Panel 2: Windowed LSF
        # Panel 3: FFT Spectrum  | Panel 4: Normalized MTF Curve
        fig_frame = tk.Frame(right_frame, bg="#181a1f")
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(7.8, 4.2), dpi=100, facecolor="#181a1f")
        self.ax_esf = self.fig.add_subplot(221)
        self.ax_lsf = self.fig.add_subplot(222)
        self.ax_fft = self.fig.add_subplot(223)
        self.ax_mtf = self.fig.add_subplot(224)

        for ax in (self.ax_esf, self.ax_lsf, self.ax_fft, self.ax_mtf):
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8a99a8", labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

        self.ax_esf.set_title("1. Normalized ESF (50/50 Symmetric)", color="#e6edf3", fontsize=9, fontweight="bold")
        self.ax_lsf.set_title("2. Line Spread Function (LSF Derivative)", color="#e6edf3", fontsize=9, fontweight="bold")
        self.ax_fft.set_title("3. FFT Magnitude Spectrum", color="#e6edf3", fontsize=9, fontweight="bold")
        self.ax_mtf.set_title("4. Final Normalized MTF (0% - 100%)", color="#e6edf3", fontsize=9, fontweight="bold")

        self.fig.tight_layout()

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Results Table
        table_frame = tk.Frame(right_frame, bg="#181a1f")
        table_frame.pack(fill=tk.X, padx=10, pady=10)

        columns = ("ROI", "Static Bal", "AI 50/50", "Static MTF", "Dynamic MTF", "PINN Neural", "Shift", "Consensus")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        widths = (75, 95, 95, 85, 95, 95, 65, 110)
        for col, w in zip(columns, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill=tk.X)

        self.tree.bind("<<TreeviewSelect>>", self._on_select_roi_row)

    def _on_select_image(self):
        f = filedialog.askopenfilename(
            title="Select Camera Test Image",
            filetypes=[("Image Files", "*.png *.bmp *.jpg *.jpeg *.tif")]
        )
        if f:
            self.current_image_path = f
            self.current_image_gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            self.btn_run.config(state=tk.NORMAL)
            self._display_image(f)

    def _display_image(self, path_or_mat):
        if isinstance(path_or_mat, str):
            mat = cv2.imread(path_or_mat)
        else:
            mat = path_or_mat

        if mat is None:
            return

        h, w = mat.shape[:2]
        cw = self.canvas_img.winfo_width() or 520
        ch = self.canvas_img.winfo_height() or 520

        scale = min(cw / w, ch / h, 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(mat, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if resized.ndim == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        im_pil = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(im_pil)

        self.canvas_img.delete("all")
        self.canvas_img.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.tk_img)

    def _on_run_analysis(self):
        if not self.current_image_path:
            return

        self.status_badge.config(text="ANALYZING...", bg="#0e639c")
        threading.Thread(target=self._run_bg_analysis, daemon=True).start()

    def _run_bg_analysis(self):
        try:
            res = process_full_image_static_vs_dynamic(self.current_image_path)
            # Add PINN neural evaluations
            for sec, comp in res["comparisons"].items():
                if comp.get("dynamic_crop") is not None:
                    neural_res = self.neural_engine.predict_neural_mtf(comp["dynamic_crop"], target_freq_lpmm=50.0)
                    comp["neural_res"] = neural_res
            self.current_results = res
            self.after(0, self._render_results)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))

    def _render_results(self):
        if not self.current_results:
            return

        res = self.current_results
        self._update_overlay_display()

        # Update Badge
        self.status_badge.config(text="50/50 OPTIMAL", bg="#2da44e")

        # Populate Table
        for row in self.tree.get_children():
            self.tree.delete(row)

        for sec, comp in res["comparisons"].items():
            s_b = comp["static_balance"]
            d_b = comp["dynamic_balance"]
            s_m = comp["static_mtf"]["MTFPercent"] if comp["static_mtf"] else 0.0
            d_m = comp["dynamic_mtf"]["MTFPercent"] if comp["dynamic_mtf"] else 0.0
            n_res = comp.get("neural_res")
            n_m = n_res["NeuralMTFPercent"] if n_res else d_m
            sh = comp["correction_shift_px"]

            delta = abs(d_m - n_m)
            consensus = "CONSENSUS_PASS" if delta <= 6.0 else "ANOMALY_WARNING"

            self.tree.insert("", tk.END, values=(
                sec,
                f"{s_b['dark_pct']:.0f}/{s_b['bright_pct']:.0f}%",
                f"{d_b['dark_pct']:.0f}/{d_b['bright_pct']:.0f}%",
                f"{min(100.0, s_m):.2f}%",
                f"{min(100.0, d_m):.2f}%",
                f"{min(100.0, n_m):.2f}%",
                f"{sh:.1f} px",
                consensus
            ))

        # Select and plot first ROI
        if self.tree.get_children():
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)
            self._plot_roi_physics_pipeline(self.tree.item(first_item)["values"][0])

    def _on_change_overlay_mode(self):
        self.overlay_mode = self.mode_var.get()
        if self.current_results and self.current_image_gray is not None:
            self._update_overlay_display()

    def _update_overlay_display(self):
        if not self.current_results or self.current_image_gray is None:
            return
        overlay_mat = generate_dual_overlay_image(
            self.current_image_gray,
            self.current_results["comparisons"],
            mode=self.overlay_mode
        )
        self._display_image(overlay_mat)

    def _on_select_roi_row(self, event):
        sel = self.tree.selection()
        if sel:
            sec_name = self.tree.item(sel[0])["values"][0]
            self._plot_roi_physics_pipeline(sec_name)

    def _plot_roi_physics_pipeline(self, sec_name):
        if not self.current_results:
            return
        comp = self.current_results["comparisons"].get(sec_name)
        if not comp:
            return

        s_mtf = comp["static_mtf"]
        d_mtf = comp["dynamic_mtf"]
        n_res = comp.get("neural_res")

        # Clear all 4 subplots
        for ax in (self.ax_esf, self.ax_lsf, self.ax_fft, self.ax_mtf):
            ax.clear()
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8a99a8", labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

        # ---------------------------------------------------------
        # 1. Panel 1: Normalized ESF (0.0 to 1.0)
        # ---------------------------------------------------------
        if s_mtf and "NormalizedESF" in s_mtf:
            self.ax_esf.plot(s_mtf["NormalizedESF"], color="#ff4444", linestyle="--", label="Static ESF (Off-Center)", alpha=0.8)
        if d_mtf and "NormalizedESF" in d_mtf:
            self.ax_esf.plot(d_mtf["NormalizedESF"], color="#00ffff", linewidth=2.0, label="Dynamic 50/50 ESF")
        self.ax_esf.set_title(f"1. Normalized ESF (ROI '{sec_name}')", color="#e6edf3", fontsize=9, fontweight="bold")
        self.ax_esf.set_xlabel("Super-Sampled Bins (0.25 px)", color="#8a99a8", fontsize=8)
        self.ax_esf.set_ylabel("Norm. Intensity (0 - 1)", color="#8a99a8", fontsize=8)
        self.ax_esf.set_ylim(-0.05, 1.05)
        self.ax_esf.legend(facecolor="#181a1f", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        # ---------------------------------------------------------
        # 2. Panel 2: Line Spread Function (LSF Derivative)
        # ---------------------------------------------------------
        if s_mtf and "NormalizedLSF" in s_mtf:
            self.ax_lsf.plot(s_mtf["NormalizedLSF"], color="#ff4444", linestyle="--", label="Static LSF", alpha=0.8)
        if d_mtf and "NormalizedLSF" in d_mtf:
            self.ax_lsf.plot(d_mtf["NormalizedLSF"], color="#00ffff", linewidth=2.0, label="Dynamic 50/50 LSF")
        self.ax_lsf.set_title("2. Line Spread Function (LSF Bell Curve)", color="#e6edf3", fontsize=9, fontweight="bold")
        self.ax_lsf.set_xlabel("Subpixel Spatial Distance", color="#8a99a8", fontsize=8)
        self.ax_lsf.set_ylabel("Amplitude (0 to 1)", color="#8a99a8", fontsize=8)
        self.ax_lsf.set_ylim(-0.05, 1.05)
        self.ax_lsf.legend(facecolor="#181a1f", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        # ---------------------------------------------------------
        # 3. Panel 3: FFT Magnitude Spectrum
        # ---------------------------------------------------------
        if d_mtf and "FFTSpectrum" in d_mtf:
            fft_v = d_mtf["FFTSpectrum"]
            n_show = min(64, len(fft_v))
            self.ax_fft.plot(fft_v[:n_show], color="#a371f7", linewidth=2.0, label="|FFT(LSF)| (DC=1.0)")
        self.ax_fft.set_title("3. Smooth FFT Magnitude Spectrum", color="#e6edf3", fontsize=9, fontweight="bold")
        self.ax_fft.set_xlabel("FFT Frequency Bins", color="#8a99a8", fontsize=8)
        self.ax_fft.set_ylabel("Magnitude (DC=1.0)", color="#8a99a8", fontsize=8)
        self.ax_fft.set_ylim(0, 1.05)
        self.ax_fft.legend(facecolor="#181a1f", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        # ---------------------------------------------------------
        # 4. Panel 4: Final Normalized MTF Curve (0% - 100%)
        # ---------------------------------------------------------
        if s_mtf:
            f_s = s_mtf["Frequencies"]
            m_s = np.clip(s_mtf["MTFCurve"], 0.0, 1.0)
            mask_s = f_s <= 100.0
            self.ax_mtf.plot(f_s[mask_s], m_s[mask_s], color="#ff4444", linestyle="--", label=f"Static MTF: {min(100.0, s_mtf['MTFPercent']):.1f}%", alpha=0.8)

        if d_mtf:
            f_d = d_mtf["Frequencies"]
            m_d = np.clip(d_mtf["MTFCurve"], 0.0, 1.0)
            mask_d = f_d <= 100.0
            self.ax_mtf.plot(f_d[mask_d], m_d[mask_d], color="#00ffff", linewidth=2.2, label=f"AI Dynamic: {min(100.0, d_mtf['MTFPercent']):.1f}%")

        if n_res:
            f_n = n_res["Frequencies"]
            m_n = np.clip(n_res["NeuralMTFCurve"], 0.0, 1.0)
            mask_n = f_n <= 100.0
            self.ax_mtf.plot(f_n[mask_n], m_n[mask_n], color="#f0883e", linewidth=1.8, linestyle=":", label=f"PINN Neural: {min(100.0, n_res['NeuralMTFPercent']):.1f}%")

        self.ax_mtf.axvline(x=50.0, color="#8a99a8", linestyle=":", label="50 lp/mm")
        self.ax_mtf.set_title("4. Normalized MTF Curve (0% - 100%)", color="#e6edf3", fontsize=9, fontweight="bold")
        self.ax_mtf.set_xlabel("Spatial Frequency (lp/mm)", color="#8a99a8", fontsize=8)
        self.ax_mtf.set_ylabel("MTF (0.0 to 1.0)", color="#8a99a8", fontsize=8)
        self.ax_mtf.set_ylim(0, 1.05)
        self.ax_mtf.set_xlim(0, 100)
        self.ax_mtf.legend(facecolor="#181a1f", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

        self.fig.tight_layout()
        self.canvas_fig.draw()


if __name__ == "__main__":
    app = DualEngineApp()
    app.mainloop()
