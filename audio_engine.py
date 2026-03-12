import queue
import numpy as np
import sounddevice as sd
from scipy.signal import windows as sig_windows
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

FFT_SIZE = 8192
N_SMOOTH = 200
F_MIN = 30.0
F_MAX = 16000.0


def log_smooth(freqs: np.ndarray, mag_db: np.ndarray,
               n_points: int = N_SMOOTH,
               f_min: float = F_MIN,
               f_max: float = F_MAX,
               sigma: float = 6.0):
    """
    Interpolate FFT bins onto a log-spaced frequency grid and apply
    Gaussian smoothing (~1/4 octave). Returns (out_freqs, out_db).
    """
    mask = (freqs >= f_min) & (freqs <= f_max) & (freqs > 0)
    if mask.sum() < 2:
        return np.logspace(np.log10(f_min), np.log10(f_max), n_points), np.full(n_points, -90.0)

    log_in = np.log10(freqs[mask])
    log_out = np.linspace(np.log10(f_min), np.log10(f_max), n_points)

    interp = interp1d(log_in, mag_db[mask], kind='linear', fill_value='extrapolate')
    out_db = interp(log_out)
    out_db = gaussian_filter1d(out_db, sigma=sigma)

    return 10.0 ** log_out, out_db


class AudioEngine:
    def __init__(self):
        self._stream = None
        self._buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        self._window = sig_windows.hann(FFT_SIZE)
        self._freqs = None
        self._channel_idx = 0
        self.sample_rate = 48000
        self.data_queue: queue.Queue = queue.Queue(maxsize=20)

    def get_devices(self):
        """Return list of (sd_index, name, max_input_channels) for input-capable devices.
        Prefers ASIO devices when available; otherwise deduplicates by name."""
        hostapis = sd.query_hostapis()
        asio_index = next(
            (i for i, h in enumerate(hostapis) if 'asio' in h['name'].lower()), None
        )

        all_devs = list(sd.query_devices())
        result = []

        if asio_index is not None:
            for i, dev in enumerate(all_devs):
                if dev['hostapi'] == asio_index and dev['max_input_channels'] > 0:
                    result.append((i, dev['name'], int(dev['max_input_channels'])))

        if not result:
            # Fallback: deduplicate by name, keep first occurrence
            seen = set()
            for i, dev in enumerate(all_devs):
                if dev['max_input_channels'] > 0 and dev['name'] not in seen:
                    seen.add(dev['name'])
                    result.append((i, dev['name'], int(dev['max_input_channels'])))

        return result

    def start(self, device_index: int, channel_idx: int = 0, sample_rate: int = 48000):
        self.stop()
        self.sample_rate = sample_rate
        self._channel_idx = channel_idx
        self._buffer = np.zeros(FFT_SIZE, dtype=np.float32)
        self._freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)

        dev_info = sd.query_devices(device_index)
        n_ch = min(int(dev_info['max_input_channels']), channel_idx + 1)

        self._stream = sd.InputStream(
            device=device_index,
            channels=n_ch,
            samplerate=sample_rate,
            blocksize=1024,
            dtype='float32',
            callback=self._callback,
            latency='low',
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time_info, status):
        ch = min(self._channel_idx, indata.shape[1] - 1)
        samples = indata[:, ch]
        self._buffer = np.roll(self._buffer, -frames)
        self._buffer[-frames:] = samples

        fft_mag = np.abs(np.fft.rfft(self._buffer * self._window))
        fft_db = 20.0 * np.log10(fft_mag / FFT_SIZE + 1e-10)

        try:
            self.data_queue.put_nowait((self._freqs, fft_db))
        except queue.Full:
            pass
