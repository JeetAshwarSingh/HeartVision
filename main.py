"""
rPPG (Remote Photoplethysmography) — Live Camera
=================================================
Implements the Plane-Orthogonal-to-Skin (POS) algorithm
(Wang et al., IEEE TBIOM, 2017) on a live webcam feed.
Stops automatically when EITHER condition is met:
  1. 60 seconds have elapsed (hard timeout)
  2. Quasi-periodicity of the cardiac signal is detected early:
       — spectral SNR in the HR band exceeds SNR_THRESHOLD, AND
       — the last STABILITY_WINDOW HR estimates agree within STABILITY_TOLERANCE bpm

Face detection uses OpenCV Haar cascade by default.
If you have OpenFace installed, set USE_OPENFACE = True and
point OPENFACE_BIN to your FeatureExtraction binary.

Requirements:
    pip install opencv-python numpy scipy pandas

Usage:
    python rppg_live.py
    Press Q to quit early.
    
"""

import os
import io
import base64
import time
import subprocess
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from scipy.signal import butter, detrend, filtfilt
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CAMERA_INDEX         = 0       # 0 = default webcam; change if you have multiple
WINDOW_SEC           = 10      # sliding signal window in seconds
MAX_DURATION_SEC     = 60      # hard stop after this many seconds
HR_LOW_HZ            = 0.75    # 45 bpm  — lower physiological bound
HR_HIGH_HZ           = 3.33    # 200 bpm — upper physiological bound

# Quasi-periodicity detection ──────────────────────────────────────────────────
# The signal is considered quasi-periodic (cardiac origin confirmed) when:
#   (a) spectral SNR >= SNR_THRESHOLD  →  a sharp dominant peak exists in the HR band
#   (b) the last STABILITY_WINDOW HR readings have std dev <= STABILITY_TOLERANCE bpm
# Both (a) and (b) must hold simultaneously.
MIN_COLLECT_SEC      = 10      # seconds — don't check quasi-periodicity before this
SNR_THRESHOLD        = 3.5     # peak-magnitude / mean-magnitude in HR band
STABILITY_WINDOW     = 5       # number of consecutive estimates to compare
STABILITY_TOLERANCE  = 5.0     # bpm — max std dev of those estimates
HR_UPDATE_EVERY_SEC  = 0.5     # seconds between HR / SNR recomputes (reduces CPU load)

# OpenFace
# Replace the path below with the output of `pwd` run inside your OpenFace dir.
USE_OPENFACE    = True
OPENFACE_BIN    = "/Users/jeetashwar/OpenFace/exe/FeatureExtraction"
OPENFACE_TMPDIR = os.path.join(os.path.expanduser("~"), ".rppg_of_tmp")
OF_ROI_REFRESH_SEC = 1.0


#SIGNAL PROCESSING 

def bandpass_filter(signal: np.ndarray, fps: float) -> np.ndarray:

    """4th-order Butterworth bandpass clamped to the physiological HR band."""
    nyq    = fps / 2.0
    low_n  = max(HR_LOW_HZ  / nyq, 1e-4)
    high_n = min(HR_HIGH_HZ / nyq, 1 - 1e-4)
    b, a   = butter(4, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def pos_wang(rgb_signal: np.ndarray, fps: float) -> np.ndarray:

    """
    Plane-Orthogonal-to-Skin (POS) algorithm.
    Wang, W. et al. (2017). IEEE Transactions on Biomedical Engineering.
    Args:
        rgb_signal : (N, 3) — mean RGB value per frame
        fps        : frames per second
    Returns:
        Detrended and bandpass-filtered rPPG waveform, shape (N,)
    """

    C = rgb_signal.T.astype(float) # (3, N)
    mean_C = np.mean(C, axis=1, keepdims=True)
    mean_C[mean_C == 0] = 1e-9
    C_n = C / mean_C - 1.0

    # POS projection: S = [[0, 1, -1], [-2, 1, 1]]

    X     = C_n[1] - C_n[2]
    Y     = C_n[1] + C_n[2] - 2.0 * C_n[0]
    std_y = np.std(Y)
    alpha = np.std(X) / (std_y if std_y > 1e-9 else 1e-9)
    h     = detrend(X + alpha * Y)

    if len(h) >= int(fps * 2):
        h = bandpass_filter(h, fps)
    return h


def estimate_hr_and_snr(waveform: np.ndarray, fps: float) -> tuple[float, float]:

    """
    FFT-based HR estimation with spectral SNR.
    SNR = peak_magnitude / mean_magnitude in the HR band.
    A clean quasi-periodic cardiac signal gives SNR >> 1 (typically 3–10+).
    Pure noise or motion artefacts give SNR ≈ 1
    Returns:
        (heart_rate_bpm, snr)
    """

    n       = len(waveform)
    freqs   = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(waveform * np.hanning(n)))

    mask = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)

    if not np.any(mask):
        return 0.0, 0.0

    mag_band  = fft_mag[mask]
    peak_idx  = np.argmax(mag_band)
    peak_freq = freqs[mask][peak_idx]
    snr       = mag_band[peak_idx] / (np.mean(mag_band) + 1e-9)
    return round(peak_freq * 60.0, 1), round(float(snr), 2)


def is_quasi_periodic(hr_history: deque, snr: float, elapsed: float) -> bool:
    """
    Returns True when BOTH conditions hold:
      (a) SNR >= SNR_THRESHOLD  (sharp spectral peak → periodic signal)
      (b) recent HR estimates have converged (std <= STABILITY_TOLERANCE bpm)

    Also requires MIN_COLLECT_SEC of data to have elapsed first.
    """
    if elapsed < MIN_COLLECT_SEC:
        return False
    if len(hr_history) < STABILITY_WINDOW:
        return False
    recent = list(hr_history)[-STABILITY_WINDOW:]
    stable = float(np.std(recent)) <= STABILITY_TOLERANCE
    return snr >= SNR_THRESHOLD and stable


#FACE DETECTION 

_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_haar(frame: np.ndarray):
    """Returns (x, y, w, h) of largest face, or None."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _haar.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


#OPENFACE INTEGRATION (optional)
# Cached forehead bbox from the last successful OpenFace call.
# Stores (y1, y2, x1, x2) in pixel coordinates.

_of_cache: dict = {"bbox": None, "last_call": 0.0}
def _landmarks_to_forehead_bbox(xs: np.ndarray, ys: np.ndarray,
                                  frame_w: int) -> tuple[int, int, int, int]:
    
    """
    Converts 68-point OpenFace landmarks to a forehead bounding box.
    Landmarks 17-26 are the two eyebrow arcs; the forehead sits above them.
    Returns (y1, y2, x1, x2).
    """

    brow_y     = int(np.min(ys[17:27]))
    brow_x_min = int(np.min(xs[17:27]))
    brow_x_max = int(np.max(xs[17:27]))
    fh_h       = int((brow_x_max - brow_x_min) * 0.4)

    y1 = max(0, brow_y - fh_h)
    x1 = max(0, brow_x_min)
    x2 = min(frame_w, brow_x_max)
    return y1, brow_y, x1, x2


def roi_from_openface(frame: np.ndarray) -> np.ndarray | None:
    """
    Returns the forehead ROI using OpenFace facial landmarks.
    Caching strategy: the subprocess call (~30-50 ms startup each time) is
    only made once every OF_ROI_REFRESH_SEC.  For all frames in between, the
    cached (y1, y2, x1, x2) crop is applied directly to the live frame
    this is near-zero cost and still gives a landmark-accurate forehead ROI
    because the face doesn't move significantly within one second.
    Falls back to None (→ Haar) if the binary is missing or the call fails.
    """

    if not OPENFACE_BIN or not os.path.exists(OPENFACE_BIN):
        return None

    now = time.time()

    #Return cached ROI if it's fresh enough

    if (  _of_cache["bbox"] is not None
          and now - _of_cache["last_call"] < OF_ROI_REFRESH_SEC):
        y1, y2, x1, x2 = _of_cache["bbox"]
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    #Refresh: call FeatureExtraction on the current frame
    os.makedirs(OPENFACE_TMPDIR, exist_ok=True)
    img_path = os.path.join(OPENFACE_TMPDIR, "of_frame.png")
    cv2.imwrite(img_path, frame)

    try:
        subprocess.run(
            [OPENFACE_BIN,
             "-f",        img_path,
             "-out_dir",  OPENFACE_TMPDIR,
             "-2Dfp",                       # write 2-D facial landmarks
             "-quiet"],
            capture_output=True, timeout=3, check=True,
        )
    except Exception:
        if _of_cache["bbox"] is not None:
            y1, y2, x1, x2 = _of_cache["bbox"]
            roi = frame[y1:y2, x1:x2]
            return roi if roi.size > 0 else None
        return None

    csv_path = os.path.join(OPENFACE_TMPDIR, "of_frame.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        import pandas as pd
        lm = pd.read_csv(csv_path)
        lm.columns = lm.columns.str.strip()

        x_cols = [c for c in lm.columns if c.startswith("x_")]
        y_cols = [c for c in lm.columns if c.startswith("y_")]
        xs = lm[x_cols].values[0].astype(int)
        ys = lm[y_cols].values[0].astype(int)

        y1, y2, x1, x2 = _landmarks_to_forehead_bbox(xs, ys, frame.shape[1])
        _of_cache["bbox"]      = (y1, y2, x1, x2)
        _of_cache["last_call"] = now

        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    except Exception:
        return None


#ROI → MEAN RGB

def get_mean_rgb(frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """
    Returns (mean_rgb (3,), bbox) using OpenFace forehead or Haar fallback.
    """
    roi, bbox = None, None

    if USE_OPENFACE:
        roi = roi_from_openface(frame)

    if roi is None:
        bbox = detect_face_haar(frame)
        if bbox is None:
            return None, None
        x, y, w, h = bbox
        roi = frame[y : y + h, x : x + w]

    if roi is None or roi.size == 0:
        return None, None

    roi_rgb  = cv2.cvtColor(cv2.resize(roi, (64, 64)), cv2.COLOR_BGR2RGB)
    mean_rgb = np.mean(roi_rgb.reshape(-1, 3), axis=0)
    return mean_rgb, bbox


#DRAWING HELPERS

def draw_timer_bar(frame: np.ndarray, elapsed: float) -> None:
    """Draws a 30-second countdown bar across the bottom of the frame."""
    H, W  = frame.shape[:2]
    frac  = min(elapsed / MAX_DURATION_SEC, 1.0)
    bar_h = 10
    y0    = H - bar_h

    # background
    cv2.rectangle(frame, (0, y0), (W, H), (50, 50, 50), -1)
    # filled portion (green → blue → red as time runs out)
    filled_w = int(W * frac)
    r = int(255 * frac)
    g = int(255 * (1.0 - frac))
    cv2.rectangle(frame, (0, y0), (filled_w, H), (0, g, r), -1)


def draw_snr_bar(frame: np.ndarray, snr: float) -> None:
    """Small horizontal SNR indicator in the top-right corner."""
    H, W    = frame.shape[:2]
    bar_w   = 120
    bar_h   = 10
    x0      = W - bar_w - 10
    y0      = 10
    frac    = min(snr / (SNR_THRESHOLD * 2), 1.0)
    filled  = int(bar_w * frac)
    color   = (0, 220, 0) if snr >= SNR_THRESHOLD else (0, 180, 220)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + bar_h), color, -1)
    cv2.putText(frame, f"SNR {snr:.1f}", (x0, y0 + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def show_final_result(frame: np.ndarray, hr_bpm: float, stop_reason: str,
                      elapsed: float) -> None:
    
    """
    Overlays a final result banner on the frame and holds it for 3 seconds.
    """
    overlay = frame.copy()
    H, W    = frame.shape[:2]

    # semi-transparent dark panel
    cv2.rectangle(overlay, (0, H // 3), (W, 2 * H // 3), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    hr_text = f"Heart Rate: {hr_bpm:.1f} bpm" if hr_bpm > 0 else "Heart Rate: N/A"
    cv2.putText(frame, hr_text, (W // 2 - 180, H // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 100), 3)
    cv2.putText(frame, stop_reason, (W // 2 - 160, H // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    cv2.putText(frame, f"Elapsed: {elapsed:.1f}s", (W // 2 - 80, H // 2 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    cv2.imshow("rPPG — Live (Q to quit)", frame)
    cv2.waitKey(3000)


#REPORT GENERATION

def _fig_to_b64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_report(
    session_ts    : str,
    stop_reason   : str,
    elapsed       : float,
    fps           : float,
    frame_count   : int,
    face_hit_count: int,
    detector_mode : str,
    hr_timeline   : list,      # [(t_sec, bpm), ...]
    snr_timeline  : list,      # [(t_sec, snr), ...]
    rgb_log       : list,      # [(t_sec, R, G, B), ...]
    final_waveform: np.ndarray | None,
    final_hr      : float,
    final_snr     : float,
) -> str:
    """
    Build a self-contained HTML report with embedded matplotlib plots and
    session statistics.  Returns the path of the saved .html file.
    """

    DARK_BG  = "#0f1117"
    CARD_BG  = "#1a1d27"
    ACCENT   = "#00e676"
    TEXT     = "#e0e0e0"
    SUBTEXT  = "#9e9e9e"
    RED      = "#ff5252"
    YELLOW   = "#ffd740"
    BLUE     = "#40c4ff"

    plt.rcParams.update({
        "figure.facecolor"  : CARD_BG,
        "axes.facecolor"    : CARD_BG,
        "axes.edgecolor"    : "#333344",
        "axes.labelcolor"   : TEXT,
        "xtick.color"       : SUBTEXT,
        "ytick.color"       : SUBTEXT,
        "text.color"        : TEXT,
        "grid.color"        : "#2a2d3a",
        "grid.linestyle"    : "--",
        "grid.alpha"        : 0.5,
        "lines.linewidth"   : 1.8,
        "font.family"       : "monospace",
    })

    plots: dict[str, str] = {}

    #Plot 1: Heart rate over time
    if hr_timeline:
        t_hr  = [p[0] for p in hr_timeline]
        v_hr  = [p[1] for p in hr_timeline]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t_hr, v_hr, color=ACCENT, linewidth=2)
        ax.fill_between(t_hr, v_hr, alpha=0.12, color=ACCENT)
        if len(v_hr) > 1:
            ax.axhline(float(np.mean(v_hr)), color=YELLOW, linewidth=1,
                       linestyle="--", label=f"Mean {np.mean(v_hr):.1f} bpm")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heart Rate (bpm)")
        ax.set_title("Heart Rate Over Time", fontsize=11, pad=8)
        ax.grid(True)
        plots["hr"] = _fig_to_b64(fig)
        plt.close(fig)

    #Plot 2: Spectral SNR over time
    if snr_timeline:
        t_snr = [p[0] for p in snr_timeline]
        v_snr = [p[1] for p in snr_timeline]
        fig, ax = plt.subplots(figsize=(9, 2.5))
        ax.plot(t_snr, v_snr, color=BLUE, linewidth=1.8)
        ax.axhline(SNR_THRESHOLD, color=RED, linewidth=1,
                   linestyle="--", label=f"Threshold ({SNR_THRESHOLD})")
        ax.fill_between(t_snr, v_snr, alpha=0.10, color=BLUE)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spectral SNR")
        ax.set_title("Signal Quality (SNR) Over Time", fontsize=11, pad=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True)
        plots["snr"] = _fig_to_b64(fig)
        plt.close(fig)

    #Plot 3: Final rPPG waveform
    if final_waveform is not None and len(final_waveform) > 0:
        t_wave = np.linspace(0, len(final_waveform) / fps, len(final_waveform))
        fig, ax = plt.subplots(figsize=(9, 2.5))
        ax.plot(t_wave, final_waveform, color="#b39ddb", linewidth=1.4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_title("rPPG Waveform — Final 10-second Window", fontsize=11, pad=8)
        ax.grid(True)
        plots["wave"] = _fig_to_b64(fig)
        plt.close(fig)

    #Plot 4: RGB channels over time
    if rgb_log:
        t_rgb = [p[0] for p in rgb_log]
        r_ch  = [p[1] for p in rgb_log]
        g_ch  = [p[2] for p in rgb_log]
        b_ch  = [p[3] for p in rgb_log]
        fig, ax = plt.subplots(figsize=(9, 2.5))
        ax.plot(t_rgb, r_ch, color="#ef5350", linewidth=1.2, label="R", alpha=0.85)
        ax.plot(t_rgb, g_ch, color="#66bb6a", linewidth=1.2, label="G", alpha=0.85)
        ax.plot(t_rgb, b_ch, color="#42a5f5", linewidth=1.2, label="B", alpha=0.85)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean pixel value")
        ax.set_title("Raw RGB Signal from Face ROI", fontsize=11, pad=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True)
        plots["rgb"] = _fig_to_b64(fig)
        plt.close(fig)

    #Derived statistics
    hr_values = [p[1] for p in hr_timeline] if hr_timeline else []
    hr_mean   = float(np.mean(hr_values))   if hr_values else 0.0
    hr_std    = float(np.std(hr_values))    if hr_values else 0.0
    hr_min    = float(np.min(hr_values))    if hr_values else 0.0
    hr_max    = float(np.max(hr_values))    if hr_values else 0.0
    face_pct  = 100.0 * face_hit_count / max(frame_count, 1)

    def stat_row(label, value, unit=""):
        return (
            f'<tr><td style="color:{SUBTEXT};padding:6px 16px 6px 0">{label}</td>'
            f'<td style="color:{ACCENT};font-weight:bold">{value}{unit}</td></tr>'
        )

    def plot_block(key, title):
        if key not in plots:
            return ""
        return (
            f'<div class="card"><h3>{title}</h3>'
            f'<img src="data:image/png;base64,{plots[key]}" '
            f'style="width:100%;border-radius:6px"></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>rPPG Session Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background:{DARK_BG}; color:{TEXT}; font-family: 'Courier New', monospace;
          padding: 32px; }}
  h1   {{ color:{ACCENT}; font-size:1.6rem; margin-bottom:4px; }}
  h2   {{ color:{TEXT}; font-size:1.1rem; margin:28px 0 12px;
          border-bottom:1px solid #2a2d3a; padding-bottom:6px; }}
  h3   {{ color:{SUBTEXT}; font-size:0.85rem; margin-bottom:10px;
          text-transform:uppercase; letter-spacing:1px; }}
  .subtitle {{ color:{SUBTEXT}; font-size:0.85rem; margin-bottom:28px; }}
  .grid    {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .card    {{ background:{CARD_BG}; border:1px solid #2a2d3a; border-radius:10px;
              padding:20px; }}
  .card.full {{ grid-column: 1 / -1; }}
  table    {{ border-collapse:collapse; width:100%; }}
  .hr-big  {{ font-size:2.6rem; color:{ACCENT}; font-weight:bold;
              letter-spacing:2px; margin:10px 0 4px; }}
  .tag     {{ display:inline-block; background:#1e2a1e; color:{ACCENT};
              border:1px solid {ACCENT}; border-radius:4px;
              padding:2px 10px; font-size:0.78rem; margin-top:6px; }}
  .tag.warn {{ background:#2a1e1e; color:{YELLOW}; border-color:{YELLOW}; }}
</style>
</head>
<body>

<h1>rPPG Session Report</h1>
<p class="subtitle">Generated {session_ts} &nbsp;|&nbsp; POS Algorithm (Wang et al., 2017)</p>

<div class="grid">

  <div class="card">
    <h3>Final Heart Rate</h3>
    <div class="hr-big">{final_hr:.1f} <span style="font-size:1.2rem;color:{SUBTEXT}">bpm</span></div>
    <div style="color:{SUBTEXT};font-size:0.85rem">Spectral SNR: <span style="color:{BLUE}">{final_snr:.2f}</span></div>
    <div class="tag">{'LOCKED' if stop_reason == 'Cardiac quasi-periodicity detected' else 'TIMEOUT'}</div>
  </div>

  <div class="card">
    <h3>Session Summary</h3>
    <table>
      {stat_row("Stop reason",    stop_reason)}
      {stat_row("Duration",       f"{elapsed:.1f}", " s")}
      {stat_row("Camera FPS",     f"{fps:.1f}")}
      {stat_row("Total frames",   frame_count)}
      {stat_row("Face detected",  f"{face_pct:.1f}", "%")}
      {stat_row("Detector",       detector_mode)}
    </table>
  </div>

  <div class="card">
    <h3>HR Statistics</h3>
    <table>
      {stat_row("Mean",   f"{hr_mean:.1f}", " bpm")}
      {stat_row("Std dev",f"{hr_std:.1f}",  " bpm")}
      {stat_row("Min",    f"{hr_min:.1f}",  " bpm")}
      {stat_row("Max",    f"{hr_max:.1f}",  " bpm")}
      {stat_row("Readings", len(hr_values))}
    </table>
  </div>

  <div class="card">
    <h3>Algorithm Config</h3>
    <table>
      {stat_row("Window",              f"{WINDOW_SEC}", " s")}
      {stat_row("HR band",             f"{HR_LOW_HZ}–{HR_HIGH_HZ}", " Hz")}
      {stat_row("SNR threshold",       SNR_THRESHOLD)}
      {stat_row("Stability window",    STABILITY_WINDOW)}
      {stat_row("Stability tolerance", STABILITY_TOLERANCE, " bpm")}
    </table>
  </div>

</div>

<h2>Plots</h2>
<div class="grid">
  {plot_block("hr",   "Heart Rate Over Time")}
  {plot_block("snr",  "Spectral SNR Over Time")}
  <div class="card full">{plot_block("wave", "rPPG Waveform — Final Window") if "wave" in plots else ""}</div>
  <div class="card full">{plot_block("rgb",  "Raw RGB Signal") if "rgb" in plots else ""}</div>
</div>

</body>
</html>
"""

    out_path = f"rppg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


#MAIN LOOP

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {CAMERA_INDEX}. "
            "Change CAMERA_INDEX in the config block."
        )

    for _ in range(5):          # warm up auto-exposure
        cap.read()

    fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
    win_size      = int(WINDOW_SEC * fps)
    update_every  = max(1, int(HR_UPDATE_EVERY_SEC * fps))

    rgb_buffer : deque = deque(maxlen=win_size)
    hr_history : deque = deque(maxlen=STABILITY_WINDOW * 3)

    hr_bpm          = 0.0
    snr             = 0.0
    frame_count     = 0
    face_hit_count  = 0
    stop_reason     = ""
    last_frame      = None
    final_waveform  = None

    # per-session logs for the report

    hr_timeline  : list = []   # [(elapsed_sec, bpm)]
    snr_timeline : list = []   # [(elapsed_sec, snr)]
    rgb_log      : list = []   # [(elapsed_sec, R, G, B)] — sampled every 0.1 s
    _last_rgb_log = 0.0

    start_time  = time.time()
    prev_time   = start_time

    mode_str = (
        f"OpenFace ({'found' if os.path.exists(OPENFACE_BIN) else 'NOT found → Haar fallback'})"
        if USE_OPENFACE else "Haar Cascade"
    )
    print(f"Camera  : index {CAMERA_INDEX}  |  {fps:.1f} FPS")
    print(f"Window  : {WINDOW_SEC}s  ({win_size} frames)")
    print(f"Timeout : {MAX_DURATION_SEC}s")
    print(f"Detector: {mode_str}")
    print("Press Q to quit early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            stop_reason = "Frame grab failed"
            break

        now         = time.time()
        elapsed     = now - start_time
        display_fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time   = now
        frame_count += 1
        last_frame   = frame.copy()

        if elapsed >= MAX_DURATION_SEC:
            stop_reason = "60-second timeout reached"
            break

        mean_rgb, bbox = get_mean_rgb(frame)
        if mean_rgb is not None:
            rgb_buffer.append(mean_rgb)
            face_hit_count += 1
            if elapsed - _last_rgb_log >= 0.1:
                rgb_log.append((elapsed, mean_rgb[0], mean_rgb[1], mean_rgb[2]))
                _last_rgb_log = elapsed

        if len(rgb_buffer) == win_size and frame_count % update_every == 0:
            waveform       = pos_wang(np.array(rgb_buffer), fps)
            hr_bpm, snr    = estimate_hr_and_snr(waveform, fps)
            if hr_bpm > 0:
                hr_history.append(hr_bpm)
                hr_timeline.append((elapsed, hr_bpm))
                snr_timeline.append((elapsed, snr))
                final_waveform = waveform

        # ── Quasi-periodicity check ────────────────────────────────────────
        if is_quasi_periodic(hr_history, snr, elapsed):
            stop_reason = "Cardiac quasi-periodicity detected"
            break

        # ── Draw bounding box ─────────────────────────────────────────────
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

        # ── Overlay text ──────────────────────────────────────────────────
        buf_pct     = int(100 * len(rgb_buffer) / win_size)
        remaining   = max(0.0, MAX_DURATION_SEC - elapsed)
        locked      = is_quasi_periodic(hr_history, snr, elapsed)

        hr_color    = (0, 255, 100) if locked else (0, 220, 0)
        hr_label    = f"HR: {hr_bpm:.1f} bpm" if hr_bpm > 0 else "HR: collecting..."
        face_label  = "Face: detected" if mean_rgb is not None else "Face: searching..."
        lock_label  = "Signal LOCKED" if locked else ""

        cv2.putText(frame, hr_label,
                    (10, 38),  cv2.FONT_HERSHEY_SIMPLEX, 1.1, hr_color, 2)
        cv2.putText(frame, f"Buffer: {buf_pct}%",
                    (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 0), 2)
        cv2.putText(frame, face_label,
                    (10, 96),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 210), 2)
        cv2.putText(frame, f"Time left: {remaining:.1f}s",
                    (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 180, 140), 2)
        cv2.putText(frame, f"FPS: {display_fps:.1f}",
                    (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        if lock_label:
            cv2.putText(frame, lock_label,
                        (10, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)

        draw_timer_bar(frame, elapsed)
        draw_snr_bar(frame, snr)

        cv2.imshow("rPPG — Live (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_reason = "Quit by user"
            break

    #Final result screen
    elapsed_total = time.time() - start_time
    if last_frame is not None and stop_reason != "Frame grab failed":
        show_final_result(last_frame, hr_bpm, stop_reason, elapsed_total)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nStopped  : {stop_reason}")
    print(f"Elapsed  : {elapsed_total:.1f}s")
    if hr_bpm > 0:
        print(f"Final HR : {hr_bpm:.1f} bpm  (SNR = {snr:.2f})")
    else:
        print("Final HR : could not be estimated (insufficient signal)")

    #Generate HTML report
    print("\nGenerating report...")
    detector_mode = (
        f"OpenFace ({'found' if os.path.exists(OPENFACE_BIN) else 'Haar fallback'})"
        if USE_OPENFACE else "Haar Cascade"
    )
    report_path = generate_report(
        session_ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stop_reason    = stop_reason,
        elapsed        = elapsed_total,
        fps            = fps,
        frame_count    = frame_count,
        face_hit_count = face_hit_count,
        detector_mode  = detector_mode,
        hr_timeline    = hr_timeline,
        snr_timeline   = snr_timeline,
        rgb_log        = rgb_log,
        final_waveform = final_waveform,
        final_hr       = hr_bpm,
        final_snr      = snr,
    )
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()