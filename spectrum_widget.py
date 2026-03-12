import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

FREQ_TICKS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

Y_MIN = -100
Y_MAX = 0

DIFF_Y_MIN = -24
DIFF_Y_MAX = 24


def _freq_label(f: float) -> str:
    return f'{int(f // 1000)}k' if f >= 1000 else str(int(f))


def _hex_with_alpha(hex_color: str, alpha: int) -> tuple:
    """Return (r, g, b, alpha) from a '#rrggbb' string."""
    c = QColor(hex_color)
    return (c.red(), c.green(), c.blue(), alpha)


class DiffWidget(pg.PlotWidget):
    """Small zero-centred panel showing target − live difference."""

    def __init__(self, parent=None):
        super().__init__(parent=parent, background='#0a0a14')

        self.setLabel('left', 'Δ dB')
        self.getViewBox().disableAutoRange()
        self.setYRange(DIFF_Y_MIN, DIFF_Y_MAX, padding=0)
        self.setXRange(np.log10(30), np.log10(16000))
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().hideButtons()
        self.setFixedHeight(140)

        # Suppress the bottom axis labels (main spectrum shows them)
        ax_bottom = self.getAxis('bottom')
        ax_bottom.setTicks([[(np.log10(f), '') for f in FREQ_TICKS]])
        ax_bottom.setHeight(0)

        # Y-axis ticks at useful dB increments
        ax_left = self.getAxis('left')
        ax_left.setTicks([[(v, str(v)) for v in range(DIFF_Y_MIN, DIFF_Y_MAX + 1, 6)]])

        # Zero reference line
        zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#555555', width=1))
        self.addItem(zero_line)

        self._color = '#e040fb'

        # Filled area between curve and zero
        self._zeros = self.plot([], [], pen=pg.mkPen(None))  # y=0 baseline
        self._curve = self.plot([], [], pen=pg.mkPen(self._color, width=2))
        self._fill = pg.FillBetweenItem(
            self._zeros, self._curve,
            brush=pg.mkBrush(_hex_with_alpha(self._color, 60)),
        )
        self.addItem(self._fill)

    def set_diff_color(self, hex_color: str):
        self._color = hex_color
        self._curve.setPen(pg.mkPen(hex_color, width=2))
        self._fill.setBrush(pg.mkBrush(_hex_with_alpha(hex_color, 60)))

    def update_diff(self, freqs: np.ndarray, diff_db: np.ndarray):
        log_x = np.log10(freqs)
        self._zeros.setData(log_x, np.zeros_like(diff_db))
        self._curve.setData(log_x, diff_db)

    def clear_diff(self):
        self._zeros.setData([], [])
        self._curve.setData([], [])


class SpectrumWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, background='#0f0f1a')

        self.setLabel('bottom', 'Frequency (Hz)')
        self.setLabel('left', 'Level (dB)')
        self.getViewBox().disableAutoRange()
        self.setYRange(Y_MIN, Y_MAX, padding=0)
        self.setXRange(np.log10(30), np.log10(16000))
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().hideButtons()

        ax = self.getAxis('bottom')
        ax.setTicks([[(np.log10(f), _freq_label(f)) for f in FREQ_TICKS]])

        self._mode = 'line'

        self.addLegend(offset=(10, 10))

        # Line plot (default)
        self._live = self.plot(pen=pg.mkPen('#00e5ff', width=2), name='Live')

        # Bar graph (hidden until mode switched)
        self._live_bars = pg.BarGraphItem(
            x=[], y0=[], y1=[], width=0.01,
            brush=pg.mkBrush('#00e5ff'),
            pen=pg.mkPen(None),
            name='Live',
        )
        self.addItem(self._live_bars)
        self._live_bars.setVisible(False)

        self._target = self.plot(
            pen=pg.mkPen('#ff9800', width=2, style=Qt.PenStyle.DashLine),
            name='Target',
        )

    def set_live_mode(self, mode: str):
        """Switch live display between 'line' and 'bars'."""
        self._mode = mode
        self._live.setVisible(mode == 'line')
        self._live_bars.setVisible(mode == 'bars')

    def update_live(self, freqs: np.ndarray, db: np.ndarray):
        if self._mode == 'bars':
            log_x = np.log10(freqs)
            spacing = (log_x[-1] - log_x[0]) / max(len(log_x) - 1, 1)
            self._live_bars.setOpts(
                x=log_x,
                y0=np.full_like(db, float(Y_MIN)),
                y1=db,
                width=spacing * 0.9,
            )
        else:
            self._live.setData(np.log10(freqs), db)
        self.setYRange(Y_MIN, Y_MAX, padding=0)

    def set_live_color(self, hex_color: str):
        self._live.setPen(pg.mkPen(hex_color, width=2))
        self._live_bars.setOpts(brush=pg.mkBrush(hex_color))

    def set_target_color(self, hex_color: str):
        self._target.setPen(pg.mkPen(hex_color, width=2, style=Qt.PenStyle.DashLine))

    def set_target(self, freqs: np.ndarray, db: np.ndarray):
        self._target.setData(np.log10(freqs), db)

    def clear_target(self):
        self._target.setData([], [])
