import numpy as np
import json
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from biosppy.signals import ecg
import pandas as pd

# Load JSON file
json_file = "D:\Work\Final Trimester\Capstone 3 (EEE 4903)\Code\Patient 2.json"
with open(json_file, 'r') as f:
    data = json.load(f)

# Load CSV file
df = pd.read_csv("Patient 2.csv")

# Convert CSV columns to numpy arrays
lead1 = np.array(df["lead1"])
lead2 = np.array(df["lead2"])
lead3 = np.array(df["lead3"])

# Stack arrays to create a 2D array representing all channels
signal_samples = np.vstack([lead1, lead2, lead3])

# Define filter parameters for fetal ECG
fs_fetal = 800  # Sampling frequency for fetal ECG
lowcut_fetal = 10  # Hz
highcut_fetal = 80  # Hz123
order = 2  # Filter order

# Apply bandpass filter to fetal ECG
nyquist_fetal = 0.5 * fs_fetal
low_fetal = lowcut_fetal / nyquist_fetal
high_fetal = highcut_fetal / nyquist_fetal
b_fetal, a_fetal = signal.butter(order, [low_fetal, high_fetal], btype="band")
filtered_fetal = signal.filtfilt(b_fetal, a_fetal, lead1)

# Define filter parameters for maternal ECG
fs_maternal = 400  # Sampling frequency for maternal ECG
lowcut_maternal = 5  # Hz
highcut_maternal = 40  # Hz

# Apply bandpass filter to maternal ECG
nyquist_maternal = 0.5 * fs_maternal
low_maternal = lowcut_maternal / nyquist_maternal
high_maternal = highcut_maternal / nyquist_maternal
b_maternal, a_maternal = signal.butter(order, [low_maternal, high_maternal], btype="band")
filtered_maternal = signal.filtfilt(b_maternal, a_maternal, lead2)

# Perform Fast ICA on fetal ECG
ica_fetal = FastICA(n_components=1)
fetal_ecg = ica_fetal.fit_transform(filtered_fetal.reshape(-1, 1))

# Perform Fast ICA on maternal ECG
ica_maternal = FastICA(n_components=1)
maternal_ecg = ica_maternal.fit_transform(filtered_maternal.reshape(-1, 1))

# Assign separated signals to variables
separated_signals = np.hstack([fetal_ecg, maternal_ecg])

# Define time axes 't_fetal' and 't_maternal'
n_samples_fetal = len(fetal_ecg)
t_fetal = np.arange(n_samples_fetal) / fs_fetal

n_samples_maternal = len(maternal_ecg)
t_maternal = np.arange(n_samples_maternal) / fs_maternal

# Define the time limit
t_max = 10  # Maximum time in seconds

# Calculate the corresponding maximum index
idx_max_fetal = int(t_max * fs_fetal)
idx_max_maternal = int(t_max * fs_maternal)

# Plot original and separated signals
plt.figure(figsize=(12, 12))

# Plot the original Lead 1 signal
plt.subplot(5, 1, 1)
plt.plot(t_fetal[:idx_max_fetal], lead1[:idx_max_fetal])
plt.title("Original Lead 1")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the original Lead 2 signal
plt.subplot(5, 1, 2)
plt.plot(t_maternal[:idx_max_maternal], lead2[:idx_max_maternal])
plt.title("Original Lead 2")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Plot the original Lead 3 signal
plt.subplot(5, 1, 3)
plt.plot(t_fetal[:idx_max_fetal], lead3[:idx_max_fetal])
plt.title("Original Lead 3")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Plot the first separated component (Fetal ECG)
plt.subplot(5, 1, 4)
plt.plot(t_fetal[:idx_max_fetal], fetal_ecg[:idx_max_fetal])
plt.title("Fetal ECG")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# Plot the second separated component (Maternal ECG)
plt.subplot(5, 1, 5)
plt.plot(t_maternal[:idx_max_maternal], maternal_ecg[:idx_max_maternal])
plt.title("Maternal ECG")
plt.xlabel("Time")
plt.ylabel("Amplitude")


# Peak detection for fetal and maternal ECG
fetal_peaks = ecg.hamilton_segmenter(signal=fetal_ecg.flatten(), sampling_rate=fs_fetal)['rpeaks']
maternal_peaks = ecg.hamilton_segmenter(signal=maternal_ecg.flatten(), sampling_rate=fs_maternal)['rpeaks']

# Calculate RR intervals for fetal and maternal ECG
rr_intervals_fetal = np.diff(fetal_peaks) / fs_fetal
rr_intervals_maternal = np.diff(maternal_peaks) / fs_maternal

# Calculate average heart rates for fetal and maternal ECG
average_heart_rate_fetal = 60 / np.mean(rr_intervals_fetal)
average_heart_rate_maternal = 60 / np.mean(rr_intervals_maternal)

# Calculate fetal heart rate variability (FHRV)
fhrv = np.std(rr_intervals_fetal)

# Calculate maternal heart rate variability (MHRV)
mhrv = np.std(rr_intervals_maternal)

# Sample P-wave angles for fetal and maternal ECG (in radians)
fetal_p_wave_angles = [0.2, 0.3, 0.25, 0.15, 0.18]  # Replace with fetal angles
maternal_p_wave_angles = [0.15, 0.22, 0.2, 0.18, 0.25]  # Replace with maternal angles

# Calculate the sum of fetal P-wave angles
sum_of_fetal_angles = np.sum(fetal_p_wave_angles)

# Calculate the sum of maternal P-wave angles
sum_of_maternal_angles = np.sum(maternal_p_wave_angles)

# Calculate the average fetal P-axis (in radians)
average_fetal_p_axis = sum_of_fetal_angles / len(fetal_p_wave_angles)

# Calculate the average maternal P-axis (in radians)
average_maternal_p_axis = sum_of_maternal_angles / len(maternal_p_wave_angles)

# Convert the results to degrees
average_fetal_p_axis_degrees = average_fetal_p_axis * (180 / np.pi)
average_maternal_p_axis_degrees = average_maternal_p_axis * (180 / np.pi)

# Sample QRS durations for fetal and maternal ECG (in milliseconds)
fetal_qrs_durations = [100, 110, 105, 95, 102]  # Replace with fetal QRS durations
maternal_qrs_durations = [90, 98, 92, 88, 94]   # Replace with maternal QRS durations

# Calculate the sum of fetal QRS durations
sum_of_fetal_qrs_durations = np.sum(fetal_qrs_durations)

# Calculate the sum of maternal QRS durations
sum_of_maternal_qrs_durations = np.sum(maternal_qrs_durations)

# Calculate the average fetal QRS interval (in milliseconds)
average_fetal_qrs_interval = sum_of_fetal_qrs_durations / len(fetal_qrs_durations)

# Calculate the average maternal QRS interval (in milliseconds)
average_maternal_qrs_interval = sum_of_maternal_qrs_durations / len(maternal_qrs_durations)

# Sample QT intervals for fetal and maternal ECG (in milliseconds)
fetal_qt_intervals = [360, 370, 365, 355, 362]  # Replace with fetal QT intervals
maternal_qt_intervals = [340, 348, 342, 338, 344] # Replace with maternal QT intervals

# Calculate the sum of fetal QT intervals
sum_of_fetal_qt_intervals = np.sum(fetal_qt_intervals)

# Calculate the sum of maternal QT intervals
sum_of_maternal_qt_intervals = np.sum(maternal_qt_intervals)

# Calculate the average fetal QT interval (in milliseconds)
average_fetal_qt_interval = sum_of_fetal_qt_intervals / len(fetal_qt_intervals)

# Calculate the average maternal QT interval (in milliseconds)
average_maternal_qt_interval = sum_of_maternal_qt_intervals / len(maternal_qt_intervals)

# Sample QTs intervals for fetal and maternal ECG (in milliseconds)
fetal_qts_intervals = [220, 230, 225, 215, 222]  # Replace with fetal QTs intervals
maternal_qts_intervals = [210, 218, 212, 208, 214]  # Replace with maternal QTs intervals

# Calculate the sum of fetal QTs intervals
sum_of_fetal_qts_intervals = np.sum(fetal_qts_intervals)

# Calculate the sum of maternal QTs intervals
sum_of_maternal_qts_intervals = np.sum(maternal_qts_intervals)

# Calculate the average fetal QTs interval (in milliseconds)
average_fetal_qts_interval = sum_of_fetal_qts_intervals / len(fetal_qts_intervals)

# Calculate the average maternal QTs interval (in milliseconds)
average_maternal_qts_interval = sum_of_maternal_qts_intervals / len(maternal_qts_intervals)

# Check for Atrial Fibrillation (AF) by analyzing P waves
def detect_af(ecg_signal, sampling_rate):
    # Detect R-peaks
    rpeaks = ecg.hamilton_segmenter(ecg_signal, sampling_rate=sampling_rate)['rpeaks']

    # Calculate the time intervals between R-peaks (RR intervals)
    rr_intervals = np.diff(rpeaks) / sampling_rate

    # Calculate the average RR interval
    avg_rr_interval = np.mean(rr_intervals)

    # Check for AF based on RR interval irregularity
    if np.std(rr_intervals) > 0.1 * avg_rr_interval:
        return "Atrial Fibrillation (P wave absent)"
    else:
        return "No Atrial Fibrillation (P wave present)"

# Detect AF for both fetal and maternal ECG
af_fetal_comment = detect_af(separated_signals[:, 0], sampling_rate=fs_fetal)
af_maternal_comment = detect_af(separated_signals[:, 1], sampling_rate=fs_maternal)

# Fetal Report
print("Average Fetal Heart Rate (BPM):", average_heart_rate_fetal)
print("Fetal Heart Rate Variability (FHRV):", fhrv)
print("Average Fetal P-Axis (Degrees):", average_fetal_p_axis_degrees)
print("Average Fetal QRS Interval (ms):", average_fetal_qrs_interval)
print("Average Fetal QT Interval (ms):", average_fetal_qt_interval)
print("Average Fetal QTs Interval (ms):", average_fetal_qts_interval)
print("Fetal Atrial Fibrillation Comment:", af_fetal_comment)
print("Fetal Heart Rate Variability (FHRV):", fhrv)

# Maternal ECG
print("Average Maternal Heart Rate (BPM):", average_heart_rate_maternal)
print("Maternal Heart Rate Variability (MHRV):", mhrv)
print("Average Maternal P-Axis (Degrees):", average_maternal_p_axis_degrees)
print("Average Maternal QRS Interval (ms):", average_maternal_qrs_interval)
print("Average Maternal QT Interval (ms):", average_maternal_qt_interval)
print("Average Maternal QTs Interval (ms):", average_maternal_qts_interval)
print("Maternal Atrial Fibrillation Comment:", af_maternal_comment)
print("Maternal Heart Rate Variability (MHRV):", mhrv)

plt.tight_layout()
plt.show()
