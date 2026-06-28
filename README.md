# HeartVision

# rPPG Live Heart Rate Estimation using the POS Algorithm

A real-time **Remote Photoplethysmography (rPPG)** system that estimates heart rate from a standard webcam using the **Plane-Orthogonal-to-Skin (POS)** algorithm proposed by **Wang et al. (2017)**. The system performs live facial analysis, extracts subtle skin color variations, estimates the Blood Volume Pulse (BVP), and computes heart rate without any physical contact.

Unlike many demonstration projects that simply display an estimated heart rate, this implementation also evaluates **signal quality** using **Spectral Signal-to-Noise Ratio (SNR)** and introduces **quasi-periodicity detection** to determine when the cardiac signal is sufficiently reliable for reporting a stable heart rate.

---

# Features

* Real-time heart rate estimation using any webcam
* Implementation of the POS (Plane-Orthogonal-to-Skin) algorithm
* OpenFace-based forehead Region of Interest (ROI)
* Automatic fallback to OpenCV Haar Cascade when OpenFace is unavailable
* Real-time FFT-based heart rate estimation
* Spectral Signal-to-Noise Ratio (SNR) estimation
* Quasi-periodicity detection for reliable signal locking
* Live overlays including:

  * Heart Rate
  * Signal Quality (SNR)
  * Face Detection
  * Buffer Status
  * Remaining Time
  * FPS
* Automatic HTML report generation
* Interactive plots of physiological signals
* Dark themed report with embedded graphs
* Automatic termination when:

  * Stable cardiac signal is detected
  * Maximum recording duration is reached
* No wearable sensors required

---

# Algorithm Overview

The pipeline follows the POS algorithm proposed by Wang et al.

```
Webcam
    │
    ▼
Face Detection
    │
    ▼
Forehead ROI Extraction
(OpenFace / Haar Cascade)
    │
    ▼
Mean RGB Extraction
    │
    ▼
RGB Signal Buffer
    │
    ▼
POS Projection
    │
    ▼
Detrending
    │
    ▼
Butterworth Bandpass Filter
    │
    ▼
FFT
    │
    ▼
Heart Rate Estimation
    │
    ▼
Spectral SNR
    │
    ▼
Quasi-Periodicity Detection
    │
    ▼
Heart Rate Lock
```

---

# How It Works

## 1. Face Detection

The system first attempts to locate facial landmarks using **OpenFace**.

If OpenFace is unavailable, it automatically switches to an OpenCV Haar Cascade detector.

This ensures the application remains usable even without OpenFace installed.

---

## 2. Forehead ROI Selection

Once the face is detected, the forehead is selected as the Region of Interest because it typically contains:

* minimal facial movement
* limited expression changes
* strong pulsatile skin signal
* fewer motion artifacts

---

## 3. RGB Signal Extraction

Each frame is resized and converted to RGB.

The average Red, Green and Blue intensity values are computed for every frame.

These RGB values form three temporal signals.

---

## 4. POS Algorithm

The Plane-Orthogonal-to-Skin algorithm normalizes RGB channels and projects them into a space that suppresses illumination changes while amplifying blood volume variations.

The implementation follows the algorithm introduced in:

> Wang W., den Brinker A.C., Stuijk S., de Haan G.
> **Algorithmic Principles of Remote PPG**
> IEEE Transactions on Biomedical Engineering, 2017.

---

## 5. Signal Processing

The extracted waveform undergoes:

* Detrending
* Fourth-order Butterworth Bandpass Filtering
* Physiological frequency isolation (0.75–3.33 Hz)

This corresponds to approximately:

**45–200 BPM**

---

## 6. Heart Rate Estimation

The processed waveform is transformed into the frequency domain using the Fast Fourier Transform (FFT).

The dominant spectral peak within the physiological heart-rate band is selected as the estimated heart rate.

---

## 7. Spectral Signal-to-Noise Ratio

Signal quality is estimated using

```
SNR = Peak Magnitude / Mean Magnitude
```

Higher SNR indicates:

* cleaner physiological signal
* lower motion artifacts
* more reliable heart rate estimation

---

## 8. Quasi-Periodicity Detection

Instead of immediately reporting the first detected peak, the system verifies that:

* spectral SNR exceeds a predefined threshold
* recent heart-rate estimates remain stable

Only after both conditions are satisfied is the signal considered physiologically reliable.

This significantly reduces unstable measurements.

---

# Live Interface

During execution the application displays:

* Current Heart Rate
* Face Detection Status
* Buffer Progress
* Remaining Time
* FPS
* Spectral SNR
* Signal Lock Status
* Live Countdown Bar

---

# Example Images

## Live Detection

```
images/live_detection.png
```

> Webcam interface while the system is collecting physiological information before heart rate estimation.

---

## Final Heart Rate

```
images/final_result.png
```

> Final result after quasi-periodicity has been detected and a stable heart rate has been estimated.

---

## Heart Rate & Spectral SNR

```
images/hr_snr_plot.png
```

Shows

* Heart Rate over Time
* Spectral SNR over Time

These plots illustrate convergence toward a stable physiological signal.

---

## rPPG Waveform

```
images/rppg_waveform.png
```

Final 10-second processed Blood Volume Pulse (BVP) waveform obtained from the POS algorithm.

---

## Raw RGB Signal

```
images/raw_rgb_signal.png
```

Temporal RGB intensity values extracted from the forehead Region of Interest.

These raw signals form the basis for the POS projection.

---

# Automatic HTML Report

After every recording session an HTML report is generated automatically.

The report includes:

* Session Summary
* Final Heart Rate
* Signal Quality
* Heart Rate Statistics
* Detector Information
* Configuration Parameters
* Heart Rate Timeline
* Spectral SNR Timeline
* Raw RGB Signal
* Final rPPG Waveform

---

# Project Structure

```
.
├── images/
│   ├── live_detection.png
│   ├── final_result.png
│   ├── hr_snr_plot.png
│   ├── rppg_waveform.png
│   └── raw_rgb_signal.png
│
├── reports/
│   └── generated HTML reports
│
├── main.py
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/rppg-live-pos.git
```

Install dependencies

```bash
pip install opencv-python numpy scipy pandas matplotlib
```

---

# OpenFace Installation (Optional)

To use landmark-based forehead extraction:

1. Install OpenFace.
2. Set

```python
USE_OPENFACE = True
```

3. Update

```python
OPENFACE_BIN = "/path/to/FeatureExtraction"
```

If OpenFace is unavailable, the system automatically falls back to Haar Cascade face detection.

---

# Running

```bash
python main.py
```

Press **Q** at any time to terminate the session.

---

# Configuration

The following parameters can be modified inside the source code.

| Parameter           | Description                       |
| ------------------- | --------------------------------- |
| WINDOW_SEC          | Sliding signal window             |
| MAX_DURATION_SEC    | Maximum recording duration        |
| HR_LOW_HZ           | Lower heart-rate frequency        |
| HR_HIGH_HZ          | Upper heart-rate frequency        |
| SNR_THRESHOLD       | Minimum acceptable signal quality |
| STABILITY_WINDOW    | Consecutive HR estimates          |
| STABILITY_TOLERANCE | Allowed HR variation              |
| USE_OPENFACE        | Enable landmark detector          |

---

# Reported Performance (Literature)

The implementation follows the **POS algorithm** proposed by Wang et al. (2017).

Reported benchmark performance on the **UBFC-rPPG** dataset from later comparative studies:

| Dataset   | Algorithm | MAE (BPM) ↓ | RMSE (BPM) ↓ | Pearson r ↑ |
| --------- | --------- | ----------: | -----------: | ----------: |
| UBFC-rPPG | POS       |    **2.44** |     **6.61** |    **0.94** |

**Note:** These values are reported in the literature for the original POS algorithm and are **not measurements of this implementation**.

---

# Limitations

* Performance decreases under rapid head motion.
* Strong illumination changes may affect signal quality.
* Dark environments reduce estimation accuracy.
* Webcam frame rate influences measurement precision.
* This project is intended for research and educational purposes only and should not be used as a medical device.

---

# Future Improvements

* MediaPipe Face Mesh integration
* GPU acceleration
* Respiratory rate estimation
* HRV analysis
* Multi-person support
* Deep-learning-based rPPG algorithms
* Skin segmentation
* Real-time SpO₂ estimation
* Cross-platform desktop application

---

# Citation

If you use this project in your research, please cite:

```bibtex
@article{wang2017pos,
  title={Algorithmic Principles of Remote PPG},
  author={Wang, Wei and den Brinker, Albertus C. and Stuijk, Sander and de Haan, Gerard},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2017},
  volume={64},
  number={7},
  pages={1479--1491}
}
```

---

# Acknowledgements

* Wang et al. for the POS algorithm.
* OpenFace for landmark-based face analysis.
* OpenCV for computer vision utilities.
* NumPy, SciPy and Matplotlib for scientific computing and visualization.

---

# License

This project is intended for academic, educational and research purposes.
Please refer to the repository license for usage terms.
