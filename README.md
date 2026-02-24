# ECG Analysis with Artificial Intelligence

This project builds on **Open-ECG-Digitizer** and adds a workflow for:

- digitizing printed 12-lead ECGs from photos,
- visualizing the reconstructed signal,
- converting the result to **WFDB** (`.dat` / `.hea`),
- and running a basic **heart-rate / rhythm analysis** (Pan–Tompkins-based).

The goal is to support ECG interpretation as a **decision-support tool** from printed ECGs (for example, prehospital ECG printouts).

---

## Table of Contents

- [Idea](#idea)
- [How it works](#how-it-works)
- [ECG Digitization](#ecg-digitization)
- [Setup](#setup)
- [How to use](#how-to-use)
  - [1) Running the digitization pipeline](#1-running-the-digitization-pipeline)
  - [2) Visualization of the extracted data](#2-visualization-of-the-extracted-data)
  - [3) Conversion to WFDB (.dat / .hea)](#3-conversion-to-wfdb-dat--hea)
  - [4) Analysis: Heart rate (Pan–Tompkins-based)](#4-analysis-heart-rate-pan--tompkins-based)
- [Project structure](#project-structure)
- [Notes and limitations](#notes-and-limitations)
- [Credits](#credits)

---

## Idea

The idea for this project came from work in prehospital emergency care.

In ambulance care, 12-lead ECGs are recorded to assess the heart's electrical activity. This helps identify time-critical conditions such as:

- acute myocardial infarction (STEMI),
- certain arrhythmias,
- and other cardiac abnormalities.

The challenge is that paramedics are not cardiologists. We are trained to recognize key findings (for example ST-segment elevation, major AV blocks, or obvious rhythm disturbances), but not with the same depth as a specialist. In real interventions, there is also limited time, stress, movement, and noise.

An AI-based tool that analyzes a photo of a printed ECG could provide useful additional clues during interventions.

> **Important:** This project is intended as a **decision-support tool**. It does **not** replace clinical judgment, medical protocols, or physician interpretation.

---

## How it works

The system uses a **multi-stage AI pipeline**.

### Input
- A photo of a printed **12-lead ECG**

### Processing stages
1. **ECG image digitization**  
   The printed waveform is converted into a machine-readable signal (time series).

2. **Signal reconstruction and export**
   - digital ECG trace visualization
   - CSV export of waveform samples
   - conversion to **WFDB** (`.dat` / `.hea`)

3. **Signal-based analysis**
   - basic measurements (e.g., heart rate via RR intervals)
   - downstream AI/rhythm/morphology models (future/optional)

### Why digitization is needed
Most ECG interpretation models expect **digitally sampled waveform data**, not photos of paper ECGs.  
Because of that, direct end-to-end interpretation from the image was not used in this implementation.

---

## ECG Digitization

For the ECG digitization step, this project uses the **Open-ECG-Digitizer** project developed by Akershus University Hospital.

### Layout adaptation (important)
The original model configuration did not work directly with our ECG printouts because the lead layout was different.

- **Original setup:** precordial leads (V1–V6) arranged vertically
- **Our setup (Lifepak 15):** **3 × 4 layout**
  - limb leads: I, II, III, aVR, aVL, aVF
  - precordial leads: V1–V6

Because of this mismatch, the configuration files were adapted so the model can correctly detect and assign leads in the printed format used in this project.

---

## Setup

The pipeline requires **Python 3.12**.

You can install from GitHub (or use a ZIP archive), then create a virtual environment and install dependencies.

### Windows (example)
```bash
python3.12 -m venv openecg-env
openecg-env\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS (example)

```bash
python3.12 -m venv openecg-env
source openecg-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages from `requirements.txt`.

---

## How to use

### 1) Running the digitization pipeline

Place a paper ECG image (`.jpg`) in the `input_ecg` directory, then run:

```bash
python3 -m src.digitize --config src\config\inference_wrapper.yml
```

> On Linux/macOS, use `/` instead of `\` in paths if needed.

### Output

When processing is complete, results are written to `output_data`.

This includes:

* a visualization of the digitization pipeline output (quality control),
* a CSV export of the reconstructed ECG signal data for downstream analysis.

---

### 2) Visualization of the extracted data

To inspect the reconstructed ECG signal after digitization, use the plotting script:

```bash
python3 scripts/plot_12lead.py output_data/digitalization/IMAGENAME_timeseries_canonical.csv
```

This script loads the exported canonical time-series CSV and renders it as a digital 12-lead ECG view.

### Purpose

* post-processing validation
* signal quality inspection
* detection of obvious digitization errors before downstream analysis

---

### 3) Conversion to WFDB (.dat / .hea)

Many ECG analysis pipelines require **WFDB** format:

* `.dat` → signal samples
* `.hea` → metadata (sampling rate, leads, record structure, etc.)

To convert the digitized canonical CSV to WFDB:

```bash
python3 scripts/csv_to_wfdb.py output_data/digitalization/IMAGENAME_timeseries_canonical.csv
```

This enables interoperability with ECG processing libraries and AI-based interpretation models.

---

### 4) Analysis: Heart rate (Pan–Tompkins-based)

For signal-based ECG analysis, use the rhythm detector script:

```bash
python3 scripts/ecg_rythm_detector.py output_data/wfdb/record
```

This script applies a **Pan–Tompkins-style QRS detection pipeline** and computes:

* detected beat count
* RR intervals
* heart-rate statistics (mean / min / max BPM)

---

## Project structure

```text
.
├── input_ecg/
│   └── *.jpg
├── output_data/
│   ├── digitalization/
│   │   ├── *_timeseries_canonical.csv
│   │   └── ... (quality-control outputs)
│   └── wfdb/
│       ├── record.dat
│       └── record.hea
├── scripts/
│   ├── plot_12lead.py
│   ├── csv_to_wfdb.py
│   └── ecg_rythm_detector.py
├── src/
│   ├── digitize.py
│   └── config/
│       └── inference_wrapper.yml
└── README.md
```

---

## Notes and limitations

* This project analyzes **photos of printed ECGs**, so quality depends on:

  * photo angle,
  * lighting,
  * print quality,
  * and paper layout.
* The digitizer configuration must match the ECG print layout.
* The heart-rate script provides **basic rhythm metrics** and is not a full diagnostic interpretation system.
* This tool is for **support**, not for standalone diagnosis.

---

## Credits

This project is based on:

* **Open-ECG-Digitizer** (Akershus University Hospital)

I added functions and scripts for:

* ECG visualization,
* WFDB conversion,
* and Pan–Tompkins-based heart-rate analysis.

If you use the original digitization code in research, please also follow the citation requirements from the upstream project.