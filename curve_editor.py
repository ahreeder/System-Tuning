"""
curve_editor.py — Popup dialog for manually editing a target curve.

Displays the curve with draggable log-spaced control points. Cubic spline
interpolation reconstructs the full curve in real time as points are dragged.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from curve_manager import save_curve, load_curve

# Number of draggable control points (log-spaced across the freq range)
N_CTRL = 28

# Drag tolerance in log10-frequency units (how close the mouse must be)
DRAG_THRESHOLD = 0.12


class CurveEditorDialog(QDialog):
    """Modal dialog for editing a saved target curve."""

    # Emitted when the curve is saved so the main window can reload it
    curve_saved = pyqtSignal(str)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Curve Editor — {name}")
        self.resize(900, 500)
        self.setModal(True)

        self._name = name
        self._drag_idx = None

        # Load curve
        self._orig_freqs, self._orig_db = load_curve(name)

        # Build log-spaced control point frequencies
        f_min = float(self._orig_freqs[0])
        f_max = float(self._orig_freqs[-1])
        self._ctrl_freqs = np.logspace(
            np.log10(f_min), np.log10(f_max), N_CTRL
        )

        # Initialise control point dB values from the loaded curve
        self._ctrl_db = np.interp(
            np.log10(self._ctrl_freqs),
            np.log10(self._orig_freqs),
            self._orig_db,
        )

        # Working copy of the full-resolution curve
        self._work_freqs = self._orig_freqs.copy()
        self._work_db    = self._orig_db.copy()

        self._setup_ui()
        self._update_curve()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # ── Header ───────────────────────────────────────────────────────
        header = QHBoxLayout()
        lbl = QLabel(f"Editing: <b>{self._name}</b>")
        lbl.setStyleSheet("color: #ccc; font-size: 12px;")
        header.addWidget(lbl)
        header.addStretch()
        hint = QLabel("Drag control points  •  Scroll to zoom  •  Right-drag to pan")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        header.addWidget(hint)
        layout.addLayout(header)

        # ── Plot ─────────────────────────────────────────────────────────
        pg.setConfigOptions(antialias=True)
        self._pw = pg.PlotWidget()
        self._pw.setBackground('#1a1a1a')
        self._pw.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Axes
        self._pw.setLogMode(x=True, y=False)
        self._pw.setLabel('bottom', 'Frequency', units='Hz')
        self._pw.setLabel('left', 'Level', units='dB')
        self._pw.setXRange(np.log10(self._orig_freqs[0]),
                           np.log10(self._orig_freqs[-1]))
        self._pw.setYRange(-30, 10)
        self._pw.showGrid(x=True, y=True, alpha=0.2)

        # 0 dB reference line
        zero_line = pg.InfiniteLine(pos=0, angle=0,
                                    pen=pg.mkPen('#444', width=1, style=Qt.PenStyle.DashLine))
        self._pw.addItem(zero_line)

        # Curve line
        self._curve_item = self._pw.plot(
            pen=pg.mkPen('#ff9800', width=2)
        )

        # Control points (draggable)
        self._scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen('#ffffff', width=1),
            brush=pg.mkBrush('#00e5ff'),
            hoverable=True,
            hoverBrush=pg.mkBrush('#ff9800'),
        )
        self._pw.addItem(self._scatter)

        layout.addWidget(self._pw)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        smooth_btn = QPushButton("Smooth")
        smooth_btn.setToolTip("Apply a light smoothing pass to the curve")
        smooth_btn.clicked.connect(self._on_smooth)
        btn_row.addWidget(smooth_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Restore the curve to its last saved state")
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.setStyleSheet("font-weight: bold;")
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

        # ── Install drag handler on the ViewBox ───────────────────────────
        vb = self._pw.getViewBox()
        self._orig_vb_drag = vb.mouseDragEvent
        vb.mouseDragEvent = self._vb_drag_event

    # ------------------------------------------------------------------ drag

    def _vb_drag_event(self, ev, axis=None):
        """Intercept left-button drags to move control points."""
        if ev.button() != Qt.MouseButton.LeftButton:
            self._orig_vb_drag(ev, axis)
            return

        ev.accept()
        vb   = self._pw.getViewBox()
        vpos = vb.mapSceneToView(ev.scenePos())

        if ev.isStart():
            mouse_log_f = vpos.x()   # already in log10 space (setLogMode)
            ctrl_log_f  = np.log10(self._ctrl_freqs)
            dists = np.abs(ctrl_log_f - mouse_log_f)
            nearest = int(np.argmin(dists))
            self._drag_idx = nearest if dists[nearest] < DRAG_THRESHOLD else None

        if self._drag_idx is not None:
            self._ctrl_db[self._drag_idx] = float(np.clip(vpos.y(), -60, 20))
            self._update_curve()

        if ev.isFinish():
            self._drag_idx = None

    # ------------------------------------------------------------------ curve update

    def _update_curve(self):
        """Rebuild the full-resolution curve from control points via cubic spline."""
        cs = CubicSpline(
            np.log10(self._ctrl_freqs),
            self._ctrl_db,
            bc_type='not-a-knot',
        )
        self._work_db = cs(np.log10(self._work_freqs))

        # Update plot items
        self._curve_item.setData(self._work_freqs, self._work_db)
        self._scatter.setData(
            x=self._ctrl_freqs,
            y=self._ctrl_db,
        )

    # ------------------------------------------------------------------ buttons

    def _on_smooth(self):
        """Smooth the control points with a Savitzky-Golay pass."""
        win = min(9, N_CTRL if N_CTRL % 2 == 1 else N_CTRL - 1)
        self._ctrl_db = savgol_filter(self._ctrl_db, window_length=win, polyorder=3)
        self._update_curve()

    def _on_reset(self):
        """Restore control points to the originally loaded curve."""
        self._ctrl_db = np.interp(
            np.log10(self._ctrl_freqs),
            np.log10(self._orig_freqs),
            self._orig_db,
        )
        self._update_curve()

    def _on_save(self):
        """Save the edited curve back to disk and close."""
        save_curve(self._name, self._work_freqs, self._work_db)
        self.curve_saved.emit(self._name)
        self.accept()
