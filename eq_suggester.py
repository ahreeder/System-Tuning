import numpy as np
from scipy.signal import find_peaks, savgol_filter


# Octave bands used for shelf detection
_OCTAVE_BANDS = [
    (20,   63),
    (63,   125),
    (125,  250),
    (250,  500),
    (500,  1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
    (8000, 16000),
]


def suggest_eq(freqs: np.ndarray, diff_db: np.ndarray,
               n_bands: int = 10, threshold: float = 1.0):
    """
    Analyze a difference curve (target - current, in dB) and return
    parametric EQ suggestions sorted by magnitude.

    Returns a list of dicts: {freq, gain, q, type}
    """
    if len(diff_db) < 11:
        return []

    # Light smoothing — small window preserves narrow/high-Q peaks
    win = 11 if len(diff_db) >= 11 else (len(diff_db) | 1)
    smoothed = savgol_filter(diff_db, window_length=win, polyorder=3)

    suggestions = []

    # ── Parametric peak detection ────────────────────────────────────────
    pos_peaks, _ = find_peaks(
        smoothed, height=threshold, prominence=threshold * 0.3, distance=3
    )
    neg_peaks, _ = find_peaks(
        -smoothed, height=threshold, prominence=threshold * 0.3, distance=3
    )

    for idx in pos_peaks:
        q = _calc_q(smoothed, idx, freqs)
        suggestions.append({
            'freq': freqs[idx],
            'gain': round(float(smoothed[idx]), 1),
            'q':    round(q, 2),
            'type': 'Peak',
        })

    for idx in neg_peaks:
        q = _calc_q(-smoothed, idx, freqs)
        suggestions.append({
            'freq': freqs[idx],
            'gain': round(float(smoothed[idx]), 1),
            'q':    round(q, 2),
            'type': 'Peak',
        })

    # ── Shelf detection ──────────────────────────────────────────────────
    shelves = _detect_shelves(freqs, smoothed, threshold)
    suggestions.extend(shelves)

    suggestions.sort(key=lambda x: abs(x['gain']), reverse=True)
    return suggestions[:n_bands]


def _detect_shelves(freqs: np.ndarray, diff_db: np.ndarray,
                    threshold: float) -> list:
    """
    Detect high-shelf and low-shelf correction needs by comparing octave-band
    averages against a mid-frequency reference.
    """
    # Compute per-octave mean and geometric-centre frequency
    means = []
    centers = []
    for f_lo, f_hi in _OCTAVE_BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if mask.sum() >= 2:
            means.append(float(np.mean(diff_db[mask])))
            centers.append(float(np.sqrt(f_lo * f_hi)))

    if len(means) < 5:
        return []

    means = np.array(means)
    centers = np.array(centers)

    # Reference level: median of the middle bands (~200 Hz – 2 kHz)
    mid_ref = float(np.median(means[2:6]))

    shelves = []

    # ── High shelf: top 3 octave bands (~2 kHz – 16 kHz) ────────────────
    hi_dev = means[-3:] - mid_ref
    pos_hi = int(np.sum(hi_dev > threshold))
    neg_hi = int(np.sum(hi_dev < -threshold))

    if pos_hi >= 2 or neg_hi >= 2:
        deviated = hi_dev[np.abs(hi_dev) > threshold]
        gain = round(float(np.mean(deviated)), 1)
        corner = _shelf_corner(centers, means, mid_ref, gain, search_from='top')
        shelves.append({'freq': corner, 'gain': gain, 'q': 0.7, 'type': 'Hi Shelf'})

    # ── Low shelf: bottom 2 octave bands (<125 Hz) ───────────────────────
    lo_dev = means[:2] - mid_ref
    pos_lo = int(np.sum(lo_dev > threshold))
    neg_lo = int(np.sum(lo_dev < -threshold))

    if pos_lo >= 2 or neg_lo >= 2:
        deviated = lo_dev[np.abs(lo_dev) > threshold]
        gain = round(float(np.mean(deviated)), 1)
        corner = _shelf_corner(centers, means, mid_ref, gain, search_from='bottom')
        shelves.append({'freq': corner, 'gain': gain, 'q': 0.7, 'type': 'Lo Shelf'})

    return shelves


def _shelf_corner(centers: np.ndarray, means: np.ndarray,
                  mid_ref: float, gain: float, search_from: str) -> int:
    """
    Find the corner frequency where the diff curve first deviates more than
    half the shelf gain from the mid reference.
    """
    half = abs(gain) / 2.0
    direction = 1 if gain > 0 else -1

    if search_from == 'top':
        # Scan from mid upward, find first band where deviation crosses half-gain
        for c, m in zip(centers, means):
            if c > 500 and (m - mid_ref) * direction > half:
                return int(round(c))
        return int(round(centers[-3]))  # fallback: first high band
    else:
        # Scan from mid downward, find first band where deviation crosses half-gain
        for c, m in zip(reversed(centers), reversed(means)):
            if c < 500 and (m - mid_ref) * direction > half:
                return int(round(c))
        return int(round(centers[1]))   # fallback: last low band


def _calc_q(curve: np.ndarray, peak_idx: int, freqs: np.ndarray) -> float:
    """Estimate Q from the -3 dB bandwidth of a peak."""
    peak_val = curve[peak_idx]
    half_power = peak_val - 3.0

    left = peak_idx - 1
    while left > 0 and curve[left] > half_power:
        left -= 1

    right = peak_idx + 1
    while right < len(curve) - 1 and curve[right] > half_power:
        right += 1

    bw = freqs[right] - freqs[left]
    if bw <= 0:
        return 1.41  # ~1 octave fallback

    return max(0.1, freqs[peak_idx] / bw)
