# Voice Gender Classification

This Python script classifies voice recordings by gender (male or female) using the Harmonic Product Spectrum (HPS) method. It processes WAV audio files and determines gender based on fundamental frequency analysis.

## Overview

- **Language:** Python
- **Dependencies:**
  - `numpy`
  - `scipy`

## Installation

Ensure you have Python installed, then install dependencies:

```bash
pip install numpy scipy
```

## Usage

### Classify a single audio file:

Run the script with the path to your WAV file:

```bash
python classify_voice.py your_audio.wav
```

Output:
```
M
```
or
```
K
```
("M" for male, "K" for female)

### Batch processing

To classify all `.wav` files in the current directory, ensure filenames indicate gender at the end (e.g., `voice1_M.wav`, `voice2_K.wav`):

```bash
python classify_voice.py
```

Output example:
```
voice1_M.wav: Oczek. M, Rozpoznano M
voice2_K.wav: Oczek. K, Rozpoznano K

Trafność: 100.00%  (2/2)
```

## How It Works

1. **Audio Processing**:
   - The script reads WAV files.
   - Converts stereo files to mono by taking the first channel.
   - Normalizes audio amplitude.

2. **Feature Extraction**:
   - Applies Hamming windowing to segments of audio data.
   - Calculates FFT to obtain the power spectrum.

3. **Harmonic Product Spectrum (HPS)**:
   - Enhances fundamental frequency by multiplying downsampled spectra.
   - Summarizes spectral energy in male (85–180 Hz) and female (165–255 Hz) frequency ranges.

4. **Gender Classification**:
   - Compares summed energies in both ranges.
   - Determines gender based on higher spectral energy.

