import numpy as np
from scipy.signal import find_peaks, savgol_filter


def suggest_eq(freqs: np.ndarray, diff_db: np.ndarray,
               n_bands: int = 8, threshold: float = 1.5):
    """
    Analyze a difference curve (target - current, in dB) and return
    parametric EQ suggestions sorted by magnitude.

    Returns a list of dicts: {freq, gain, q}
    """
    if len(diff_db) < 21:
        return []

    # Smooth the difference curve before peak detection
    win = 21 if len(diff_db) >= 21 else (len(diff_db) | 1)
    smoothed = savgol_filter(diff_db, window_length=win, polyorder=3)

    suggestions = []

    # Positive peaks (need boost)
    pos_peaks, _ = find_peaks(
        smoothed, height=threshold, prominence=threshold * 0.5, distance=5
    )
    # Negative peaks (need cut) — invert and find
    neg_peaks, _ = find_peaks(
        -smoothed, height=threshold, prominence=threshold * 0.5, distance=5
    )

    for idx in pos_peaks:
        q = _calc_q(smoothed, idx, freqs)
        suggestions.append({
            'freq': freqs[idx],
            'gain': round(float(smoothed[idx]), 1),
            'q': round(q, 2),
        })

    for idx in neg_peaks:
        q = _calc_q(-smoothed, idx, freqs)
        suggestions.append({
            'freq': freqs[idx],
            'gain': round(float(smoothed[idx]), 1),
            'q': round(q, 2),
        })

    suggestions.sort(key=lambda x: abs(x['gain']), reverse=True)
    return suggestions[:n_bands]


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
