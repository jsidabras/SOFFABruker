#!/usr/bin/env python3
"""
Straightforward Tk GUI for the standard SOFFA pipeline.

Pipeline mirrored from the C++ implementation:
1. Load DSC parameters and matching DTA data
2. Optional per-segment Savitzky-Golay smoothing (when SG filter is selected)
3. Build the SOFFA high-resolution overlap-add grid
4. Apply either Savitzky-Golay or Gaussian smoothing on the fine grid
5. Decimate with the SOFFA moving-average method to 512/1024/2048/4096 points

Dependencies:
  - Python standard library
  - numpy
  - matplotlib (optional, for the embedded preview plot)
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


ALLOWED_FINAL_POINTS = (512, 1024, 2048, 4096)


@dataclass
class ProcessingParams:
    filter_type: str = "Savitzky-Golay"
    sg_window: int = 5
    sg_order: int = 2
    gaussian_sigma: float = 10.0
    step_field: float = 0.5
    np_points: int = 8192
    center_field: float = 3443.0
    sweep_field: float = 30.0
    step_num: int = 260
    target_points: int = 1024
    moving_average_window: int = 10


@dataclass
class ProcessedResult:
    field_axis: np.ndarray
    spectrum: np.ndarray
    fine_grid: np.ndarray
    notes: list[str]


def parse_numeric_token(value: str) -> float:
    return float(value.strip().split()[0])


def parse_dsc(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line[0] in "*#.":
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                params[key.strip()] = value.strip()
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                params[parts[0].strip()] = parts[1].strip()
            elif len(parts) == 1:
                params[parts[0].strip()] = ""
    return params


def load_processing_params_from_dsc(dsc_path: Path) -> ProcessingParams:
    raw = parse_dsc(dsc_path)
    params = ProcessingParams()

    xmin_val = None
    xwid_val = None
    cf_val = None
    sw_val = None
    xpts_val = None
    ypts_val = None

    if "SFNP" in raw:
        params.np_points = int(parse_numeric_token(raw["SFNP"]))
    if "SFST" in raw:
        params.step_field = float(parse_numeric_token(raw["SFST"]))
    if "SFSW" in raw:
        params.sweep_field = float(parse_numeric_token(raw["SFSW"]))
    if "SFFW" in raw:
        params.gaussian_sigma = float(parse_numeric_token(raw["SFFW"]))
    if "SFTP" in raw:
        params.target_points = int(parse_numeric_token(raw["SFTP"]))
    if "SFSN" in raw:
        params.step_num = int(parse_numeric_token(raw["SFSN"]))
    if "SFCF" in raw:
        params.center_field = float(parse_numeric_token(raw["SFCF"]))
    if "SFMA" in raw:
        params.moving_average_window = int(parse_numeric_token(raw["SFMA"]))

    if "XPTS" in raw:
        xpts_val = int(parse_numeric_token(raw["XPTS"]))
    if "YPTS" in raw:
        ypts_val = int(parse_numeric_token(raw["YPTS"]))
    if "XMIN" in raw:
        xmin_val = float(parse_numeric_token(raw["XMIN"]))
    if "XWID" in raw:
        xwid_val = float(parse_numeric_token(raw["XWID"]))
    if "CenterField" in raw:
        cf_val = float(parse_numeric_token(raw["CenterField"]))
    if "SweepWidth" in raw:
        sw_val = float(parse_numeric_token(raw["SweepWidth"]))

    if "SFNP" not in raw and xpts_val is not None and xpts_val > 0:
        params.np_points = xpts_val
        params.target_points = xpts_val
    if "SFSN" not in raw and ypts_val is not None and ypts_val > 0:
        params.step_num = ypts_val
    if "SFCF" not in raw:
        if cf_val is not None:
            params.center_field = cf_val
        elif xmin_val is not None and xwid_val is not None:
            params.center_field = xmin_val + xwid_val / 2.0
    if "SFSW" not in raw:
        if sw_val is not None:
            params.sweep_field = sw_val
        elif xwid_val is not None:
            params.sweep_field = xwid_val

    if params.target_points not in ALLOWED_FINAL_POINTS:
        params.target_points = 1024

    return params


def find_dta_path(dsc_path: Path) -> Path:
    for suffix in (".DTA", ".dta"):
        candidate = dsc_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No DTA file found alongside {dsc_path.name}")


def load_dta_data(dta_path: Path) -> np.ndarray:
    return np.fromfile(dta_path, dtype=">f8").astype(np.float64)


def validate_params(params: ProcessingParams) -> None:
    if params.filter_type not in {"Savitzky-Golay", "Gaussian"}:
        raise ValueError("Filter type must be Savitzky-Golay or Gaussian.")
    if params.step_field <= 0.0:
        raise ValueError("Step size must be > 0.")
    if params.sweep_field <= 0.0:
        raise ValueError("Segment sweep width must be > 0.")
    if params.np_points < 2:
        raise ValueError("Points per segment must be at least 2.")
    if params.step_num < 1:
        raise ValueError("Number of segments must be at least 1.")
    if params.target_points not in ALLOWED_FINAL_POINTS:
        raise ValueError("Final points must be 512, 1024, 2048, or 4096.")
    if params.moving_average_window < 1:
        raise ValueError("Decimation moving-average window must be at least 1.")
    if params.sg_window < 3 or params.sg_window % 2 == 0:
        raise ValueError("Savitzky-Golay window must be odd and at least 3.")
    if params.sg_order < 0 or params.sg_order >= params.sg_window:
        raise ValueError("Savitzky-Golay order must satisfy 0 <= order < window.")
    if params.gaussian_sigma <= 0.0:
        raise ValueError("Gaussian sigma must be > 0.")


def compute_sg_coefficients(window_size: int, poly_order: int) -> np.ndarray:
    half = window_size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vstack([x ** k for k in range(poly_order + 1)]).T
    normal = design.T @ design
    target = design[half]
    coeffs = design @ np.linalg.solve(normal, target)
    return coeffs.astype(np.float64)


def apply_sg_1d(signal: np.ndarray, window_size: int, poly_order: int) -> np.ndarray:
    coeffs = compute_sg_coefficients(window_size, poly_order)
    half = window_size // 2
    out = np.empty_like(signal, dtype=np.float64)
    n = signal.size
    for i in range(n):
        start = max(0, i - half)
        stop = min(n, i + half + 1)
        coeff_start = half - (i - start)
        coeff_stop = coeff_start + (stop - start)
        chunk = signal[start:stop]
        local_coeffs = coeffs[coeff_start:coeff_stop]
        weight_sum = local_coeffs.sum()
        if weight_sum == 0.0:
            out[i] = signal[i]
        else:
            out[i] = float(np.dot(chunk, local_coeffs) / weight_sum)
    return out


def apply_per_segment_sg(
    data: np.ndarray, step_num: int, np_points: int, window_size: int, poly_order: int
) -> np.ndarray:
    matrix = data[: step_num * np_points].reshape(step_num, np_points)
    filtered = np.empty_like(matrix, dtype=np.float64)
    for step in range(step_num):
        filtered[step] = apply_sg_1d(matrix[step], window_size, poly_order)
    return filtered.reshape(-1)


def apply_gaussian_1d(signal: np.ndarray, sigma: float) -> np.ndarray:
    sigma_int = int(round(sigma))
    kernel_size = sigma_int * 6 + 1
    kernel_size = min(kernel_size, signal.size)
    if kernel_size < 1:
        raise ValueError("Gaussian kernel size is invalid.")

    half = kernel_size // 2
    x = np.arange(kernel_size, dtype=np.float64) - half
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    out = np.empty_like(signal, dtype=np.float64)
    n = signal.size
    for i in range(n):
        start = max(0, i - half)
        stop = min(n, i + (kernel_size - half))
        kernel_start = half - (i - start)
        kernel_stop = kernel_start + (stop - start)
        chunk = signal[start:stop]
        local_kernel = kernel[kernel_start:kernel_stop]
        weight_sum = local_kernel.sum()
        out[i] = 0.0 if weight_sum == 0.0 else float(np.dot(chunk, local_kernel) / weight_sum)
    return out


def create_high_resolution_grid(data: np.ndarray, params: ProcessingParams) -> np.ndarray:
    fine_points = params.target_points * 4
    first_field = params.center_field - params.sweep_field / 2.0
    last_field = first_field + (params.step_num * params.step_field)
    field_range = last_field - first_field
    field_increment = field_range / (fine_points - 1)

    matrix = data[: params.step_num * params.np_points].reshape(params.step_num, params.np_points)
    accumulator = np.zeros(fine_points, dtype=np.float64)
    counts = np.zeros(fine_points, dtype=np.int64)
    point_offsets = np.linspace(
        -params.sweep_field / 2.0,
        params.sweep_field / 2.0,
        params.np_points,
        dtype=np.float64,
    )

    for step in range(params.step_num):
        center = first_field + step * params.step_field
        fields = center + point_offsets
        mask = (fields >= first_field) & (fields <= last_field)
        if not np.any(mask):
            continue
        grid_idx = np.rint((fields[mask] - first_field) / field_increment).astype(np.int64)
        grid_idx = np.clip(grid_idx, 0, fine_points - 1)
        row = matrix[step, mask]
        accumulator += np.bincount(grid_idx, weights=row, minlength=fine_points)
        counts += np.bincount(grid_idx, minlength=fine_points)

    valid = counts > 0
    if not np.any(valid):
        raise ValueError("No data landed on the SOFFA fine grid.")

    accumulator[valid] /= counts[valid]
    valid_idx = np.flatnonzero(valid)
    accumulator = np.interp(np.arange(fine_points), valid_idx, accumulator[valid_idx])
    return accumulator


def decimate_moving_average(fine_grid: np.ndarray, params: ProcessingParams) -> tuple[np.ndarray, np.ndarray]:
    fine_points = fine_grid.size
    first_field = params.center_field - params.sweep_field / 2.0
    last_field = first_field + (params.step_num * params.step_field)
    field_axis = np.linspace(first_field, last_field, params.target_points, dtype=np.float64)

    result = np.empty(params.target_points, dtype=np.float64)
    field_range = last_field - first_field
    half_window = params.moving_average_window // 2

    for i, field in enumerate(field_axis):
        full_field_pos = 0.0 if field_range == 0.0 else (field - first_field) / field_range
        center_idx = int(full_field_pos * (fine_points - 1))
        start = max(0, center_idx - half_window)
        stop = min(fine_points, center_idx + half_window + 1)
        result[i] = float(np.mean(fine_grid[start:stop]))

    return field_axis, result


def run_standard_soffa(dsc_path: Path, params: ProcessingParams) -> ProcessedResult:
    validate_params(params)

    dta_path = find_dta_path(dsc_path)
    raw = load_dta_data(dta_path)
    required = params.step_num * params.np_points
    notes: list[str] = []

    if raw.size < required:
        raise ValueError(
            f"DTA contains {raw.size} points, but the selected geometry requires {required}."
        )
    if raw.size > required:
        notes.append(f"Input has {raw.size} values; using the first {required} values.")

    working = raw[:required].astype(np.float64, copy=True)

    if params.filter_type == "Savitzky-Golay":
        working = apply_per_segment_sg(
            working,
            params.step_num,
            params.np_points,
            params.sg_window,
            params.sg_order,
        )
        notes.append("Applied per-segment Savitzky-Golay before overlap-add.")

    fine_grid = create_high_resolution_grid(working, params)

    if params.filter_type == "Savitzky-Golay":
        fine_grid = apply_sg_1d(fine_grid, params.sg_window, params.sg_order)
        notes.append("Applied Savitzky-Golay on the fine grid.")
    else:
        fine_grid = apply_gaussian_1d(fine_grid, params.gaussian_sigma)
        notes.append("Applied Gaussian smoothing on the fine grid.")

    field_axis, spectrum = decimate_moving_average(fine_grid, params)
    return ProcessedResult(field_axis=field_axis, spectrum=spectrum, fine_grid=fine_grid, notes=notes)


def write_csv(path: Path, field_axis: np.ndarray, spectrum: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Field (G)", "Intensity"])
        for x_val, y_val in zip(field_axis, spectrum):
            writer.writerow([f"{x_val:.6f}", f"{y_val:.12e}"])


def write_dta(path: Path, spectrum: np.ndarray) -> None:
    np.asarray(spectrum, dtype=">f8").tofile(path)


def write_processed_dsc(path: Path, params: ProcessingParams, field_axis: np.ndarray) -> None:
    xmin = float(field_axis[0])
    xwid = float(field_axis[-1] - field_axis[0]) if field_axis.size > 1 else 0.0
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#DESC 1.2 * DESCRIPTOR INFORMATION ***********************\n")
        handle.write("BSEQ    BIG\n")
        handle.write("IKKF    REAL\n")
        handle.write("XTYP    IDX\n")
        handle.write("YTYP    NODATA\n")
        handle.write("ZTYP    NODATA\n")
        handle.write("IRFMT   D\n")
        handle.write(f"XPTS    {params.target_points}\n")
        handle.write(f"XMIN    {xmin:.6f}\n")
        handle.write(f"XWID    {xwid:.6f}\n")
        handle.write("IRNAM   'Intensity'\n")
        handle.write("XNAM    'Field'\n")
        handle.write("XUNI    'G'\n")
        handle.write("* SOFFA Processing Parameters:\n")
        handle.write(f"SFNP    {params.np_points}\n")
        handle.write(f"SFST    {params.step_field:.6f}\n")
        handle.write(f"SFSW    {params.sweep_field:.6f}\n")
        handle.write(f"SFFW    {params.gaussian_sigma:.6f}\n")
        handle.write(f"SFTP    {params.target_points}\n")
        handle.write(f"SFSN    {params.step_num}\n")
        handle.write(f"SFCF    {params.center_field:.6f}\n")
        handle.write(f"SFMA    {params.moving_average_window}\n")
        handle.write(f"CMNT    'Processed with standard_soffa_gui.py ({params.filter_type})'\n")


def compute_noise_region_std(
    field_axis: np.ndarray, spectrum: np.ndarray, noise_lo: float, noise_hi: float
) -> tuple[float, int]:
    """Return std over a chosen field window and the number of points used."""
    x = np.asarray(field_axis, dtype=np.float64)
    y = np.asarray(spectrum, dtype=np.float64)
    if x.size != y.size or x.size == 0:
        return float("nan"), 0
    if noise_lo >= noise_hi:
        return float("nan"), 0
    mask = np.isfinite(x) & np.isfinite(y) & (x >= noise_lo) & (x <= noise_hi)
    count = int(np.count_nonzero(mask))
    if count < 2:
        return float("nan"), count
    return float(np.std(y[mask])), count


class StandardSoffaApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Standard SOFFA GUI")
        self.root.geometry("980x760")

        self.loaded_dsc: Path | None = None
        self.result: ProcessedResult | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        file_frame = ttk.LabelFrame(outer, text="Input", padding=8)
        file_frame.pack(fill="x")

        self.path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="Browse DSC...", command=self._browse_dsc).pack(side="left", padx=(8, 0))
        ttk.Button(file_frame, text="Reload Params", command=self._reload_params).pack(side="left", padx=(8, 0))

        self.info_var = tk.StringVar(value="Load a DSC file to populate the SOFFA geometry.")
        ttk.Label(outer, textvariable=self.info_var, foreground="gray").pack(fill="x", pady=(6, 8))

        params_frame = ttk.LabelFrame(outer, text="Processing Parameters", padding=8)
        params_frame.pack(fill="x")

        self.filter_var = tk.StringVar(value="Savitzky-Golay")
        self.sg_window_var = tk.StringVar(value="5")
        self.sg_order_var = tk.StringVar(value="2")
        self.gaussian_sigma_var = tk.StringVar(value="10.0")
        self.step_field_var = tk.StringVar(value="0.5")
        self.np_points_var = tk.StringVar(value="8192")
        self.center_field_var = tk.StringVar(value="3443.0")
        self.sweep_field_var = tk.StringVar(value="30.0")
        self.step_num_var = tk.StringVar(value="260")
        self.target_points_var = tk.StringVar(value="1024")
        self.ma_window_var = tk.StringVar(value="10")

        fields = [
            ("Filter", self._build_filter_widget),
            ("S-G Window", lambda parent: ttk.Entry(parent, textvariable=self.sg_window_var, width=12)),
            ("S-G Order", lambda parent: ttk.Entry(parent, textvariable=self.sg_order_var, width=12)),
            ("Gaussian Sigma", lambda parent: ttk.Entry(parent, textvariable=self.gaussian_sigma_var, width=12)),
            ("Step Size", lambda parent: ttk.Entry(parent, textvariable=self.step_field_var, width=12)),
            ("Points / Segment", lambda parent: ttk.Entry(parent, textvariable=self.np_points_var, width=12)),
            ("Center Field", lambda parent: ttk.Entry(parent, textvariable=self.center_field_var, width=12)),
            ("Segment Sweep Width", lambda parent: ttk.Entry(parent, textvariable=self.sweep_field_var, width=12)),
            ("Segments / Steps", lambda parent: ttk.Entry(parent, textvariable=self.step_num_var, width=12)),
            ("Final Points", self._build_target_widget),
            ("Decimation MA Window", lambda parent: ttk.Entry(parent, textvariable=self.ma_window_var, width=12)),
        ]

        for idx, (label, builder) in enumerate(fields):
            row = idx // 3
            col = idx % 3
            cell = ttk.Frame(params_frame)
            cell.grid(row=row, column=col, sticky="ew", padx=6, pady=4)
            ttk.Label(cell, text=label).pack(anchor="w")
            builder(cell).pack(anchor="w")

        for col in range(3):
            params_frame.columnconfigure(col, weight=1)

        analysis_frame = ttk.LabelFrame(outer, text="Analysis", padding=8)
        analysis_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(
            analysis_frame,
            text="Signal = peak-to-peak of the processed spectrum.",
            foreground="black",
        ).pack(anchor="w", pady=(0, 4))
        noise_frame = ttk.Frame(analysis_frame)
        noise_frame.pack(fill="x")
        self.noise_lo_var = tk.StringVar(value="0.0")
        self.noise_hi_var = tk.StringVar(value="0.0")
        ttk.Label(noise_frame, text="Noise σ field range B_lo/B_hi (G):").pack(side="left")
        ttk.Entry(noise_frame, textvariable=self.noise_lo_var, width=10).pack(side="left", padx=(6, 4))
        ttk.Entry(noise_frame, textvariable=self.noise_hi_var, width=10).pack(side="left", padx=(0, 6))
        self.noise_region_var = tk.StringVar(
            value="Set B_lo < B_hi to measure noise σ in that window."
        )
        ttk.Label(noise_frame, textvariable=self.noise_region_var, foreground="gray").pack(side="left")

        action_frame = ttk.Frame(outer)
        action_frame.pack(fill="x", pady=(10, 8))

        self.process_button = ttk.Button(action_frame, text="Run Standard SOFFA", command=self._start_processing)
        self.process_button.pack(side="left")
        ttk.Button(action_frame, text="Save CSV...", command=self._save_csv).pack(side="left", padx=(8, 0))
        ttk.Button(action_frame, text="Save DTA/DSC...", command=self._save_dta_dsc).pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(outer, textvariable=self.status_var, foreground="steelblue").pack(fill="x", pady=(0, 8))

        output_frame = ttk.LabelFrame(outer, text="Output", padding=8)
        output_frame.pack(fill="both", expand=True)

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(8, 4.6), dpi=100)
            self.axes = self.figure.add_subplot(111)
            self.axes.set_xlabel("Field (G)")
            self.axes.set_ylabel("Intensity")
            self.axes.set_title("Processed spectrum preview")
            self.canvas = FigureCanvasTkAgg(self.figure, master=output_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.figure = None
            self.axes = None
            self.canvas = None
            ttk.Label(
                output_frame,
                text="matplotlib is not available, so the plot preview is disabled.",
                foreground="gray",
            ).pack(anchor="w")

        self.summary = tk.Text(output_frame, height=8, wrap="word")
        self.summary.pack(fill="x", pady=(8, 0))
        self.summary.insert("1.0", "No processed output yet.\n")
        self.summary.configure(state="disabled")

        self.filter_var.trace_add("write", self._update_filter_state)
        self._update_filter_state()

    def _build_filter_widget(self, parent: ttk.Frame) -> ttk.Widget:
        return ttk.Combobox(
            parent,
            textvariable=self.filter_var,
            values=["Savitzky-Golay", "Gaussian"],
            state="readonly",
            width=18,
        )

    def _build_target_widget(self, parent: ttk.Frame) -> ttk.Widget:
        return ttk.Combobox(
            parent,
            textvariable=self.target_points_var,
            values=[str(value) for value in ALLOWED_FINAL_POINTS],
            state="readonly",
            width=10,
        )

    def _update_filter_state(self, *_args: object) -> None:
        is_sg = self.filter_var.get() == "Savitzky-Golay"
        sg_state = "normal" if is_sg else "disabled"
        gauss_state = "disabled" if is_sg else "normal"

        self._set_entry_state(self.sg_window_var, sg_state)
        self._set_entry_state(self.sg_order_var, sg_state)
        self._set_entry_state(self.gaussian_sigma_var, gauss_state)

    def _set_entry_state(self, variable: tk.StringVar, state: str) -> None:
        for widget in self.root.winfo_children():
            self._set_entry_state_recursive(widget, variable, state)

    def _set_entry_state_recursive(self, widget: tk.Widget, variable: tk.StringVar, state: str) -> None:
        if isinstance(widget, ttk.Entry) and str(widget.cget("textvariable")) == str(variable):
            widget.configure(state=state)
        for child in widget.winfo_children():
            self._set_entry_state_recursive(child, variable, state)

    def _browse_dsc(self) -> None:
        path = filedialog.askopenfilename(
            title="Open DSC file",
            filetypes=[("Bruker DSC", "*.DSC *.dsc"), ("All files", "*.*")],
        )
        if not path:
            return
        self.path_var.set(path)
        self._load_dsc(Path(path))

    def _reload_params(self) -> None:
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Pick a DSC file first.")
            return
        self._load_dsc(Path(path))

    def _load_dsc(self, dsc_path: Path) -> None:
        try:
            params = load_processing_params_from_dsc(dsc_path)
            dta_path = find_dta_path(dsc_path)
            dta_size = load_dta_data(dta_path).size

            self.loaded_dsc = dsc_path
            self.path_var.set(str(dsc_path))
            self._set_params(params)
            self.info_var.set(
                f"Loaded {dsc_path.name} with {dta_size} DTA values. "
                f"Geometry: {params.step_num} segments x {params.np_points} points."
            )
            self.status_var.set("Parameters loaded from DSC.")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _set_params(self, params: ProcessingParams) -> None:
        self.filter_var.set(params.filter_type)
        self.sg_window_var.set(str(params.sg_window))
        self.sg_order_var.set(str(params.sg_order))
        self.gaussian_sigma_var.set(str(params.gaussian_sigma))
        self.step_field_var.set(str(params.step_field))
        self.np_points_var.set(str(params.np_points))
        self.center_field_var.set(str(params.center_field))
        self.sweep_field_var.set(str(params.sweep_field))
        self.step_num_var.set(str(params.step_num))
        target = params.target_points if params.target_points in ALLOWED_FINAL_POINTS else 1024
        self.target_points_var.set(str(target))
        self.ma_window_var.set(str(params.moving_average_window))
        self._update_filter_state()

    def _collect_params(self) -> ProcessingParams:
        return ProcessingParams(
            filter_type=self.filter_var.get().strip(),
            sg_window=int(self.sg_window_var.get().strip()),
            sg_order=int(self.sg_order_var.get().strip()),
            gaussian_sigma=float(self.gaussian_sigma_var.get().strip()),
            step_field=float(self.step_field_var.get().strip()),
            np_points=int(self.np_points_var.get().strip()),
            center_field=float(self.center_field_var.get().strip()),
            sweep_field=float(self.sweep_field_var.get().strip()),
            step_num=int(self.step_num_var.get().strip()),
            target_points=int(self.target_points_var.get().strip()),
            moving_average_window=int(self.ma_window_var.get().strip()),
        )

    def _start_processing(self) -> None:
        if self.loaded_dsc is None:
            path = self.path_var.get().strip()
            if not path:
                messagebox.showwarning("No file", "Pick a DSC file first.")
                return
            self.loaded_dsc = Path(path)

        try:
            params = self._collect_params()
            validate_params(params)
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.process_button.configure(state="disabled")
        self.status_var.set("Running standard SOFFA pipeline...")

        def worker() -> None:
            try:
                result = run_standard_soffa(self.loaded_dsc, params)
            except Exception as exc:
                self.root.after(0, lambda: self._finish_processing(error=str(exc)))
                return
            self.root.after(0, lambda: self._finish_processing(result=result))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_processing(self, result: ProcessedResult | None = None, error: str | None = None) -> None:
        self.process_button.configure(state="normal")
        if error is not None:
            self.status_var.set("Processing failed.")
            messagebox.showerror("SOFFA error", error)
            return

        assert result is not None
        self.result = result
        self.status_var.set("Processing complete.")
        self._update_plot()
        self._update_summary()

    def _update_plot(self) -> None:
        if self.result is None or self.axes is None or self.canvas is None:
            return
        self.axes.clear()
        self.axes.plot(self.result.field_axis, self.result.spectrum, color="navy", linewidth=1.2)
        noise_lo, noise_hi = self._get_noise_region()
        ptp = float(np.ptp(self.result.spectrum))
        noise_std, noise_count = compute_noise_region_std(
            self.result.field_axis, self.result.spectrum, noise_lo, noise_hi
        )
        if noise_lo < noise_hi:
            self.axes.axvspan(
                noise_lo,
                noise_hi,
                alpha=0.10,
                color="tomato",
                zorder=0,
                label=f"Noise σ window [{noise_lo:.3f}, {noise_hi:.3f}] G",
            )
        self.axes.set_xlabel("Field (G)")
        self.axes.set_ylabel("Intensity")
        if noise_lo < noise_hi and noise_count >= 2 and np.isfinite(noise_std):
            if noise_std > 0.0:
                snr_text = f"SNR = ptp / σ = {ptp / noise_std:.6g}"
            else:
                snr_text = "SNR = ptp / σ = undefined (σ = 0)"
            self.axes.set_title(f"Processed spectrum preview\n{snr_text}")
        else:
            self.axes.set_title("Processed spectrum preview\nSNR unavailable: set valid noise σ window")
        self.axes.grid(True, alpha=0.3)
        if noise_lo < noise_hi:
            self.axes.legend(loc="best", fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def _get_noise_region(self) -> tuple[float, float]:
        try:
            noise_lo = float(self.noise_lo_var.get().strip())
            noise_hi = float(self.noise_hi_var.get().strip())
        except Exception:
            return 0.0, 0.0
        if noise_lo >= noise_hi:
            return 0.0, 0.0
        return noise_lo, noise_hi

    def _update_summary(self) -> None:
        if self.result is None:
            return
        noise_lo, noise_hi = self._get_noise_region()
        noise_std, noise_count = compute_noise_region_std(
            self.result.field_axis, self.result.spectrum, noise_lo, noise_hi
        )
        ptp = float(np.ptp(self.result.spectrum))
        if noise_lo < noise_hi and noise_count >= 2 and np.isfinite(noise_std):
            self.noise_region_var.set(
                f"Using {noise_count} pts in [{noise_lo:.3f}, {noise_hi:.3f}] G for noise σ."
            )
            noise_line = f"Noise σ [{noise_lo:.6f}, {noise_hi:.6f}] G: {noise_std:.6g}"
            snr_line = (
                f"Peak-to-peak / noise σ: {ptp / noise_std:.6g}" if noise_std > 0.0 else
                "Peak-to-peak / noise σ: undefined (σ = 0)"
            )
        elif noise_lo < noise_hi:
            self.noise_region_var.set(
                f"Window [{noise_lo:.3f}, {noise_hi:.3f}] G has only {noise_count} valid pts."
            )
            noise_line = f"Noise σ [{noise_lo:.6f}, {noise_hi:.6f}] G: insufficient points"
            snr_line = "Peak-to-peak / noise σ: unavailable"
        else:
            self.noise_region_var.set("Set B_lo < B_hi to measure noise σ in that window.")
            noise_line = "Noise σ: set B_lo < B_hi"
            snr_line = "Peak-to-peak / noise σ: unavailable"
        lines = [
            f"Output points: {self.result.spectrum.size}",
            f"Field range: {self.result.field_axis[0]:.6f} G to {self.result.field_axis[-1]:.6f} G",
            f"Signal (peak-to-peak): {ptp:.6g}",
            noise_line,
            f"Metric: signal = peak-to-peak, noise = σ in the selected field window",
            snr_line,
            f"Min / Max: {np.min(self.result.spectrum):.6g} / {np.max(self.result.spectrum):.6g}",
            "",
            "Notes:",
        ]
        lines.extend(f"- {note}" for note in self.result.notes)

        self.summary.configure(state="normal")
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", "\n".join(lines) + "\n")
        self.summary.configure(state="disabled")

    def _save_csv(self) -> None:
        if self.result is None:
            messagebox.showwarning("No result", "Run the pipeline first.")
            return
        initial = "soffa_processed.csv"
        if self.loaded_dsc is not None:
            initial = f"{self.loaded_dsc.stem}_soffa.csv"
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            initialfile=initial,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        try:
            write_csv(Path(path), self.result.field_axis, self.result.spectrum)
            self.status_var.set(f"Saved CSV to {Path(path).name}.")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def _save_dta_dsc(self) -> None:
        if self.result is None:
            messagebox.showwarning("No result", "Run the pipeline first.")
            return
        if self.loaded_dsc is None:
            messagebox.showwarning("No input", "Load a DSC file first.")
            return

        initial = f"{self.loaded_dsc.stem}_soffa.DTA"
        path = filedialog.asksaveasfilename(
            title="Save processed DTA",
            defaultextension=".DTA",
            initialfile=initial,
            filetypes=[("Bruker DTA", "*.DTA"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            out_dta = Path(path)
            out_dsc = out_dta.with_suffix(".DSC")
            params = self._collect_params()
            write_dta(out_dta, self.result.spectrum)
            write_processed_dsc(out_dsc, params, self.result.field_axis)
            self.status_var.set(f"Saved {out_dta.name} and {out_dsc.name}.")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))


def main() -> None:
    root = tk.Tk()
    app = StandardSoffaApp(root)

    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists():
            app.path_var.set(str(candidate))
            app._load_dsc(candidate)

    root.mainloop()


if __name__ == "__main__":
    main()
