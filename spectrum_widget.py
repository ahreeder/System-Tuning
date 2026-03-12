import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

FREQ_TICKS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

Y_MIN = -100
Y_MAX = 0


def _freq_label(f: float) -> str:
    return f'{int(f // 1000)}k' if f >= 1000 else str(int(f))


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
        self._diff = self.plot(pen=pg.mkPen('#e040fb', width=1.5), name='Difference')

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

    def set_diff_color(self, hex_color: str):
        self._diff.setPen(pg.mkPen(hex_color, width=1.5))

    def set_target(self, freqs: np.ndarray, db: np.ndarray):
        self._target.setData(np.log10(freqs), db)

    def update_diff(self, freqs: np.ndarray, diff_db: np.ndarray):
        self._diff.setData(np.log10(freqs), diff_db)

    def clear_target(self):
        self._target.setData([], [])
        self._diff.setData([], [])
