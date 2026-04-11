# Pan-Tompkins QRS Detection with LMS Enhancement

## Overview

This project implements the Pan-Tompkins algorithm for QRS detection in ECG signals and extends it using an LMS-based adaptive threshold.

The work focuses on reproducing the original signal processing pipeline and evaluating how adaptive thresholding affects detection performance, especially in noisy conditions.

## Implementation

The algorithm follows the standard stages: bandpass filtering, derivative filtering, squaring, and moving window integration, followed by peak detection.

An additional LMS-based method is used to replace the fixed threshold with a dynamic one that adapts based on the signal.

## Data

The implementation uses ECG signals from the MIT-BIH Arrhythmia Database.
A synthetic signal is generated if the dataset is not available.

## Running the code

Run the main file:

```bash id="ab12c3"
python main.py
```

The script processes the signal, applies both detection methods, plots the intermediate stages, and prints a comparison of the results.

## Results

The adaptive method shows improved behavior in some noisy cases, while the original method remains stable for cleaner signals.

## Notes

This project was developed as part of a DSP course and focuses on understanding and analyzing the algorithm rather than optimization.

##contributors
- Raseel Jafar
- Layal Hajji
- Basmala Abu Hakema
