import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTableWidget, QTableWidgetItem,
    QInputDialog, QMessageBox, QGroupBox, QHeaderView, QStatusBar, QSpinBox,
    QDoubleSpinBox, QColorDialog,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor

from audio_engine import AudioEngine, log_smooth, N_SMOOTH, F_MIN, F_MAX
from spectrum_widget import SpectrumWidget
from curve_manager import save_curve, load_curve, list_curves
from eq_suggester import suggest_eq

EMA_ALPHA = 0.15  # default smoothing speed — overridden by UI combo

BAR_RESOLUTIONS = [
    ("1/3 Oct",  31),
    ("1/6 Oct",  61),
    ("1/12 Oct", 121),
    ("Fine",     200),
]

AVG_PRESETS = [
    ("Fast",      0.40),
    ("Medium",    0.15),
    ("Slow",      0.05),
    ("Very Slow", 0.015),
]


def _interp_to(source_freqs, source_db, target_freqs):
    """Interpolate a curve onto a different frequency grid (log domain)."""
    return np.interp(
        np.log10(target_freqs),
        np.log10(source_freqs),
        source_db,
        left=source_db[0],
        right=source_db[-1],
    )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PA Tuning Tool")
        self.resize(1200, 720)

        self._engine = AudioEngine()
        self._avg_freqs: np.ndarray | None = None
        self._avg_db: np.ndarray | None = None
        self._target_freqs: np.ndarray | None = None
        self._target_db: np.ndarray | None = None

        self._setup_ui()
        self._refresh_devices()
        self._refresh_curves()

        self._timer = QTimer()
        self._timer.setInterval(50)  # 20 fps
        self._timer.timeout.connect(self._on_timer)

    # ------------------------------------------------------------------ UI

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QHBoxLayout()

        top.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(300)
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        top.addWidget(self._device_combo)

        top.addWidget(QLabel("Ch:"))
        self._channel_combo = QComboBox()
        self._channel_combo.setMinimumWidth(70)
        top.addWidget(self._channel_combo)

        top.addWidget(QLabel("Rate:"))
        self._rate_combo = QComboBox()
        for r in [44100, 48000, 96000]:
            self._rate_combo.addItem(f'{r} Hz', r)
        self._rate_combo.setCurrentIndex(1)  # 48 kHz default
        top.addWidget(self._rate_combo)

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFixedWidth(90)
        self._start_btn.clicked.connect(self._on_start)
        top.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedWidth(90)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        top.addWidget(self._stop_btn)

        top.addWidget(QLabel("Gain:"))
        self._gain_spin = QSpinBox()
        self._gain_spin.setRange(-40, 40)
        self._gain_spin.setValue(0)
        self._gain_spin.setSuffix(" dB")
        self._gain_spin.setFixedWidth(75)
        top.addWidget(self._gain_spin)

        top.addWidget(QLabel("Avg:"))
        self._avg_combo = QComboBox()
        for label, alpha in AVG_PRESETS:
            self._avg_combo.addItem(label, alpha)
        self._avg_combo.setCurrentIndex(1)  # Medium default
        self._avg_combo.setFixedWidth(100)
        top.addWidget(self._avg_combo)

        top.addWidget(QLabel("Style:"))
        self._style_combo = QComboBox()
        self._style_combo.addItem("Line", "line")
        self._style_combo.addItem("Bars", "bars")
        self._style_combo.setFixedWidth(75)
        self._style_combo.currentIndexChanged.connect(self._on_style_changed)
        top.addWidget(self._style_combo)

        top.addWidget(QLabel("Res:"))
        self._res_combo = QComboBox()
        for label, n in BAR_RESOLUTIONS:
            self._res_combo.addItem(label, n)
        self._res_combo.setCurrentIndex(3)  # Fine default
        self._res_combo.setFixedWidth(90)
        self._res_combo.currentIndexChanged.connect(self._on_resolution_changed)
        top.addWidget(self._res_combo)

        top.addWidget(QLabel("Colors:"))
        self._live_color = '#00e5ff'
        self._target_color = '#ff9800'
        self._diff_color = '#e040fb'
        self._live_color_btn = self._make_color_btn(self._live_color)
        self._target_color_btn = self._make_color_btn(self._target_color)
        self._diff_color_btn = self._make_color_btn(self._diff_color)
        self._live_color_btn.setToolTip("Live curve color")
        self._target_color_btn.setToolTip("Target curve color")
        self._diff_color_btn.setToolTip("Difference curve color")
        self._live_color_btn.clicked.connect(self._on_live_color)
        self._target_color_btn.clicked.connect(self._on_target_color)
        self._diff_color_btn.clicked.connect(self._on_diff_color)
        top.addWidget(self._live_color_btn)
        top.addWidget(self._target_color_btn)
        top.addWidget(self._diff_color_btn)

        top.addStretch()
        root.addLayout(top)

        # ── Spectrum ──────────────────────────────────────────────────────
        self._spectrum = SpectrumWidget()
        root.addWidget(self._spectrum, stretch=1)

        # ── Bottom row ────────────────────────────────────────────────────
        bottom = QHBoxLayout()

        # Curve group
        curve_group = QGroupBox("Target Curve")
        cl = QHBoxLayout(curve_group)

        self._capture_btn = QPushButton("Capture as Target")
        self._capture_btn.clicked.connect(self._on_capture)
        cl.addWidget(self._capture_btn)

        self._load_combo = QComboBox()
        self._load_combo.setMinimumWidth(180)
        cl.addWidget(self._load_combo)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load)
        cl.addWidget(load_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear_target)
        cl.addWidget(clear_btn)

        cl.addWidget(QLabel("Offset:"))
        self._target_offset_spin = QSpinBox()
        self._target_offset_spin.setRange(-30, 30)
        self._target_offset_spin.setValue(0)
        self._target_offset_spin.setSuffix(" dB")
        self._target_offset_spin.setFixedWidth(75)
        self._target_offset_spin.valueChanged.connect(self._refresh_target_display)
        cl.addWidget(self._target_offset_spin)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(32)
        refresh_btn.clicked.connect(self._refresh_curves)
        cl.addWidget(refresh_btn)

        bottom.addWidget(curve_group)

        # EQ suggestion group
        eq_group = QGroupBox("EQ Suggestions")
        el = QVBoxLayout(eq_group)

        eq_top = QHBoxLayout()
        self._suggest_btn = QPushButton("Suggest EQ")
        self._suggest_btn.clicked.connect(self._on_suggest_eq)
        eq_top.addWidget(self._suggest_btn)

        eq_top.addWidget(QLabel("Threshold:"))
        self._eq_threshold_spin = QDoubleSpinBox()
        self._eq_threshold_spin.setRange(0.5, 6.0)
        self._eq_threshold_spin.setValue(1.0)
        self._eq_threshold_spin.setSingleStep(0.5)
        self._eq_threshold_spin.setSuffix(" dB")
        self._eq_threshold_spin.setFixedWidth(80)
        eq_top.addWidget(self._eq_threshold_spin)

        eq_top.addWidget(QLabel("Max bands:"))
        self._eq_bands_spin = QSpinBox()
        self._eq_bands_spin.setRange(1, 16)
        self._eq_bands_spin.setValue(10)
        self._eq_bands_spin.setFixedWidth(50)
        eq_top.addWidget(self._eq_bands_spin)

        eq_top.addStretch()
        el.addLayout(eq_top)

        self._eq_table = QTableWidget(0, 5)
        self._eq_table.setHorizontalHeaderLabels(['Band', 'Type', 'Freq', 'Gain (dB)', 'Q'])
        self._eq_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._eq_table.setMaximumHeight(150)
        self._eq_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._eq_table.setAlternatingRowColors(True)
        el.addWidget(self._eq_table)

        bottom.addWidget(eq_group, stretch=1)
        root.addLayout(bottom)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — select an audio device and press Start.")

    # ------------------------------------------------------------------ devices

    def _refresh_devices(self):
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._devices = self._engine.get_devices()
        for idx, name, _ in self._devices:
            self._device_combo.addItem(f'[{idx}] {name}', idx)
        self._device_combo.blockSignals(False)
        self._on_device_changed()

    def _on_device_changed(self):
        self._channel_combo.clear()
        i = self._device_combo.currentIndex()
        if i < 0 or i >= len(self._devices):
            return
        _, _, max_ch = self._devices[i]
        for ch in range(max_ch):
            self._channel_combo.addItem(f'Ch {ch + 1}', ch)

    # ------------------------------------------------------------------ start/stop

    def _on_start(self):
        i = self._device_combo.currentIndex()
        if i < 0:
            return
        device_idx = self._device_combo.currentData()
        channel_idx = self._channel_combo.currentData() or 0
        sample_rate = self._rate_combo.currentData()

        self._avg_db = None
        self._avg_freqs = None

        try:
            self._engine.start(device_idx, channel_idx, sample_rate)
        except Exception as exc:
            QMessageBox.critical(self, "Stream Error", str(exc))
            return

        self._timer.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status.showMessage(
            f"Streaming — device [{device_idx}] · ch {channel_idx + 1} · {sample_rate} Hz"
        )

    def _on_stop(self):
        self._timer.stop()
        self._engine.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped.")

    # ------------------------------------------------------------------ colors

    def _make_color_btn(self, hex_color: str) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(24, 24)
        btn.setStyleSheet(f'background-color: {hex_color}; border: 1px solid #666; border-radius: 3px;')
        return btn

    def _pick_color(self, current: str, title: str):
        color = QColorDialog.getColor(QColor(current), self, title)
        return color.name() if color.isValid() else None

    def _on_live_color(self):
        c = self._pick_color(self._live_color, "Live Curve Color")
        if c:
            self._live_color = c
            self._live_color_btn.setStyleSheet(f'background-color: {c}; border: 1px solid #666; border-radius: 3px;')
            self._spectrum.set_live_color(c)

    def _on_target_color(self):
        c = self._pick_color(self._target_color, "Target Curve Color")
        if c:
            self._target_color = c
            self._target_color_btn.setStyleSheet(f'background-color: {c}; border: 1px solid #666; border-radius: 3px;')
            self._spectrum.set_target_color(c)

    def _on_diff_color(self):
        c = self._pick_color(self._diff_color, "Difference Curve Color")
        if c:
            self._diff_color = c
            self._diff_color_btn.setStyleSheet(f'background-color: {c}; border: 1px solid #666; border-radius: 3px;')
            self._spectrum.set_diff_color(c)

    # ------------------------------------------------------------------ style / resolution

    def _on_style_changed(self):
        self._spectrum.set_live_mode(self._style_combo.currentData())

    def _on_resolution_changed(self):
        # Array size changes so the running average must restart
        self._avg_db = None
        self._avg_freqs = None

    # ------------------------------------------------------------------ timer

    def _on_timer(self):
        # Drain queue — keep only the most recent frame
        latest = None
        while not self._engine.data_queue.empty():
            try:
                latest = self._engine.data_queue.get_nowait()
            except Exception:
                break

        if latest is None:
            return

        raw_freqs, raw_db = latest
        n_pts = self._res_combo.currentData()
        smooth_freqs, smooth_db = log_smooth(raw_freqs, raw_db, n_pts, F_MIN, F_MAX)

        # Exponential moving average — speed set by UI combo
        alpha = self._avg_combo.currentData()
        if self._avg_db is None:
            self._avg_db = smooth_db.copy()
            self._avg_freqs = smooth_freqs.copy()
        else:
            self._avg_db = alpha * smooth_db + (1.0 - alpha) * self._avg_db

        gain = self._gain_spin.value()
        self._spectrum.update_live(self._avg_freqs, self._avg_db + gain)

        # Update difference curve
        if self._target_db is not None:
            target_aligned = _interp_to(self._target_freqs, self._target_db, self._avg_freqs)
            self._spectrum.update_diff(self._avg_freqs, target_aligned - self._avg_db)

    # ------------------------------------------------------------------ curves

    def _on_capture(self):
        if self._avg_db is None:
            QMessageBox.warning(self, "No Data", "Start the audio stream first.")
            return
        name, ok = QInputDialog.getText(self, "Save Target Curve", "Name for this curve:")
        if not ok or not name.strip():
            return
        name = name.strip()
        save_curve(name, self._avg_freqs, self._avg_db)
        self._target_freqs = self._avg_freqs.copy()
        self._target_db = self._avg_db.copy()
        self._refresh_curves()
        self._refresh_target_display()
        self._status.showMessage(f"Target curve '{name}' saved.")

    def _on_load(self):
        name = self._load_combo.currentText()
        if not name:
            return
        try:
            freqs, db = load_curve(name)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._target_freqs = freqs
        self._target_db = db
        self._refresh_target_display()
        self._status.showMessage(f"Target curve '{name}' loaded.")

    def _refresh_target_display(self):
        if self._target_freqs is not None:
            offset = self._target_offset_spin.value()
            self._spectrum.set_target(self._target_freqs, self._target_db + offset)

    def _on_clear_target(self):
        self._target_db = None
        self._target_freqs = None
        self._spectrum.clear_target()
        self._status.showMessage("Target curve cleared.")

    def _refresh_curves(self):
        self._load_combo.clear()
        for name in list_curves():
            self._load_combo.addItem(name)

    # ------------------------------------------------------------------ EQ suggest

    def _on_suggest_eq(self):
        if self._avg_db is None:
            QMessageBox.warning(self, "No Data", "Start the audio stream first.")
            return
        if self._target_db is None:
            QMessageBox.warning(self, "No Target", "Load a target curve first.")
            return

        target_aligned = _interp_to(self._target_freqs, self._target_db, self._avg_freqs)
        diff = target_aligned - self._avg_db
        bands = suggest_eq(
            self._avg_freqs, diff,
            n_bands=self._eq_bands_spin.value(),
            threshold=self._eq_threshold_spin.value(),
        )

        self._eq_table.setRowCount(0)
        for num, band in enumerate(bands, start=1):
            row = self._eq_table.rowCount()
            self._eq_table.insertRow(row)

            freq = band['freq']
            freq_str = f"{freq / 1000:.2f} kHz" if freq >= 1000 else f"{freq:.0f} Hz"
            gain = band['gain']
            band_type = band.get('type', 'Peak')

            self._eq_table.setItem(row, 0, QTableWidgetItem(str(num)))
            self._eq_table.setItem(row, 1, QTableWidgetItem(band_type))

            self._eq_table.setItem(row, 2, QTableWidgetItem(freq_str))

            gain_item = QTableWidgetItem(f"{gain:+.1f}")
            gain_item.setForeground(
                QColor('#4caf50') if gain > 0 else QColor('#f44336')
            )
            gain_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._eq_table.setItem(row, 3, gain_item)
            self._eq_table.setItem(row, 4, QTableWidgetItem(str(band['q'])))

        if not bands:
            self._status.showMessage("No significant differences found — curves are already close.")
        else:
            self._status.showMessage(f"EQ suggestion: {len(bands)} band(s) identified.")
