import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

FREQ_TICKS = [20, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def _freq_label(f: float) -> str:
    return f'{int(f // 1000)}k' if f >= 1000 else str(int(f))


class SpectrumWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, background='#0f0f1a')

        self.setLabel('bottom', 'Frequency (Hz)')
        self.setLabel('left', 'Level (dB)')
        self.getViewBox().disableAutoRange()
        self.setYRange(-60, 12, padding=0)
        self.setXRange(np.log10(20), np.log10(20000))
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setMouseEnabled(x=False, y=False)

        ax = self.getAxis('bottom')
        ax.setTicks([[(np.log10(f), _freq_label(f)) for f in FREQ_TICKS]])

        self.addLegend(offset=(10, 10))
        self._live = self.plot(pen=pg.mkPen('#00e5ff', width=2), name='Live')
        self._target = self.plot(
            pen=pg.mkPen('#ff9800', width=2, style=Qt.PenStyle.DashLine),
            name='Target',
        )
        self._diff = self.plot(pen=pg.mkPen('#e040fb', width=1.5), name='Difference')

    def update_live(self, freqs: np.ndarray, db: np.ndarray):
        self._live.setData(np.log10(freqs), db)
        self.setYRange(-60, 12, padding=0)

    def set_target(self, freqs: np.ndarray, db: np.ndarray):
        self._target.setData(np.log10(freqs), db)

    def update_diff(self, freqs: np.ndarray, diff_db: np.ndarray):
        self._diff.setData(np.log10(freqs), diff_db)

    def clear_target(self):
        self._target.setData([], [])
        self._diff.setData([], [])
