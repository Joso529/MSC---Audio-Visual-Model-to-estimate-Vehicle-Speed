import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import librosa
import librosa.display
from glob import glob
from math import pi
import scipy as sp
import scipy.fftpack as sf
import scipy.signal as sig
import os
import scipy.signal
import scipy.io.wavfile
audio_files = glob("C:/Users/jonat/Downloads/carnoises/RenaultScenic/RenaultScenic_101.wav")
file_number = 0
FRAME_SIZE = 2048
HOP_LENGTH = 512
base_name = audio_files[file_number].split('.')[0]
print(audio_files[file_number])
y, sr = librosa.load(audio_files[file_number],sr = 44100)
sample_duration = 1/sr
duration = sample_duration *len(y)
t1 = np.arange(0,len(y))* sample_duration
a = os.path.basename(audio_files[file_number]) # obtain last component of path
annotation_number = (os.path.splitext(a)[0])
annotations = glob(f'C:/Users/jonat/Downloads/carnoises/**/{annotation_number}.txt')
vehicle = annotation_number.split('_')[0]
f = open(annotations[0], 'r')
separate = f.read().split(' ')
speed = float(separate[0])
pass_by = float(separate[1])

print (f'sr:{sr}')
print(f'duration of signal: {duration:.2f}')
print(f'Vehicle speed is:{speed}km/h')
print(f'pass by time is:{pass_by}s')
print()
sos = scipy.signal.butter(2, 300, 'highpass', fs=sr, output='sos')
filtered_y = scipy.signal.sosfiltfilt(sos, y)
lb = librosa.feature.rms(y=filtered_y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

frames = range(0, lb.size)
t = librosa.frames_to_time(frames,hop_length=HOP_LENGTH,sr=sr)
plt.plot(t1,filtered_y)
plt.plot(t, lb,label='RMS')

plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title(f'{vehicle} signal at {speed}km/h')
#amplitude_envelope
# def amplitude_envelope(signal,frame_size,hop_length):
#     amplitude_envelope = []
#
#     for i in range(0,len(signal),hop_length):
#         current_frame_amplitude_envelope = max(signal[i:i+frame_size])
#         amplitude_envelope.append(current_frame_amplitude_envelope)
#
#     return np.array(amplitude_envelope)
#
# ae_signal = amplitude_envelope(y,FRAME_SIZE,HOP_LENGTH)
# frames = range(0, ae_signal.size)
# t = librosa.frames_to_time(frames,hop_length=HOP_LENGTH,sr=sr)
# rms_signal = librosa.feature.rms(y=y,frame_length = FRAME_SIZE,hop_length=HOP_LENGTH)[0]
# frames = range(0, rms_signal.size)
# t = librosa.frames_to_time(frames,hop_length=HOP_LENGTH,sr=sr)
# plt.plot(t, rms_signal,label='rms energy')
# # plt.xlabel("Time(s)")
# # plt.ylabel("Amplitude")
# plt.plot(t,ae_signal, label='Amplitude envelope')
#
# #rms energy
# rms_signal = librosa.feature.rms(y=y,frame_length = FRAME_SIZE,hop_length=HOP_LENGTH)[0]
# frames = range(0, rms_signal.size)
# t = librosa.frames_to_time(frames,hop_length=HOP_LENGTH,sr=sr)
#
# plt.plot(t, rms_signal,label='raw')
# plt.legend()
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title(f'RMS energy of {vehicle} signal at {speed}km/h')
# #
#
# #zero crossing rate
# zcr = librosa.feature.zero_crossing_rate(y,frame_length=FRAME_SIZE,hop_length=HOP_LENGTH)[0]
# frames = range(0, zcr.size)
# t = librosa.frames_to_time(frames,hop_length=HOP_LENGTH,sr=sr)
# plt.figure(figsize=(10,5))
# plt.plot(t, zcr)
# plt.xlabel("Time(s)")
# plt.title(f'Zero crossing rate of {vehicle} signal at {speed}km/h')
#
# #Magnitude spectrum
# ft = sp.fft.fft(y)
# magnitude = np.absolute(ft)
# frequency = np.linspace(0,sr,len(magnitude))
# half_length = len(magnitude) // 2
# frequency = frequency[:half_length]
# magnitude = magnitude[:half_length]
# plt.figure(figsize = (10,5))
# plt.plot(frequency, magnitude)
# plt.title(f'Magnitude spectrum of vehicle at {speed}km/h')
# plt.xlabel("Frequency(Hz)")
# plt.ylabel("Magnitude")
#
# #Spectrogram
# S = librosa.stft(y)
# S_db = librosa.amplitude_to_db(np.abs(S), ref = np.max)
# rms_signal = librosa.feature.rms(S = S_db, frame_length=FRAME_SIZE,hop_length=HOP_LENGTH)[0]
# frames = range(0, rms_signal.size)
# t = librosa.frames_to_time(frames,hop_length=HOP_LENGTH,sr=sr)
# plt.figure(figsize=(10,5))
# plt.plot(t, rms_signal)
#
# fig,ax = plt.subplots(figsize = (10,5))
# img = librosa.display.specshow(S_db, sr=sr,x_axis = 'time',y_axis = 'log',ax = ax)
# ax.set_title(f'Spectrogram of {vehicle} at {speed}')
# fig.colorbar(img,ax=ax, format = f'%0.2f')
# #
# #melspectrogram
# S_mel = librosa.feature.melspectrogram(y = y,sr = sr, n_mels = 512,)
# S_db_mel = librosa.amplitude_to_db(S_mel, ref=np.max)
# fig, ax = plt.subplots(figsize = (10,5))
# img = librosa.display.specshow(S_db_mel, sr=sr, x_axis= 'time', y_axis= 'log', ax = ax)
# fig.colorbar(img,ax=ax, format = f'%0.2f')
# flattened_mel = S.flatten()
#
# #spectral centroid
# sc = librosa.feature.spectral_centroid(y=y, sr=sr,n_fft=FRAME_SIZE,hop_length=HOP_LENGTH)[0]
# frames = range(len(sc))
# t = librosa.frames_to_time(frames,sr=sr, hop_length=HOP_LENGTH)
# plt.figure(figsize=(10,5))
# plt.plot(t,sc)
# plt.xlabel("Time(s)")
# plt.ylabel("Spectral Centroid(Hz)")
# plt.title(f'Spectral centroid of {vehicle} signal at {speed}km/h')
#
# #Spectral flux
# sf = librosa.onset.onset_strength(y=y, sr=sr)
# frames = range(len(sf))
# t = librosa.frames_to_time(frames,sr=sr, hop_length=HOP_LENGTH)
# plt.figure(figsize=(10,5))
# plt.plot(t,sf)
#
# #band energy ratio
# sp = librosa.stft(y,n_fft=FRAME_SIZE,hop_length=HOP_LENGTH)
# spec_t = sp.T
# def calculate_split_frequency_bin(spectrogram,split_frequency,sr):
#     frequency_range = sr/2 #spectrogram reduces the frequency range from sr to nyquist frq
#     frequency_delta_per_bin = frequency_range/spectrogram.shape[0] #how much the frequency moves
#     split_frequency_bin = np.floor(split_frequency/frequency_delta_per_bin)
#     return(int(split_frequency_bin))
# split_frequency_bin = calculate_split_frequency_bin(sp,2000,22050)
# def calculate_band_energy_ratio(sp, split_frequency, sr):
#     split_frequency_bin = calculate_split_frequency_bin(sp,split_frequency,sr)
#     power_spec = np.abs(sp) ** 2
#     power_spec = power_spec.T
#     band_energy_ratio = []
#
#     for frequencies_in_frame in power_spec:
#         sum_power_low_frequencies = np.sum(frequencies_in_frame[:split_frequency_bin])
#         sum_power_high_frequencies = np.sum(frequencies_in_frame[split_frequency_bin])
#         ber_current_frame = sum_power_low_frequencies/sum_power_high_frequencies
#         band_energy_ratio.append(ber_current_frame)
#     return np.array(band_energy_ratio)
# ber = calculate_band_energy_ratio(sp,1000,sr)
# frames = range(len(ber))
# t = librosa.frames_to_time(frames,sr=sr,hop_length=HOP_LENGTH)
# plt.figure(figsize=(10,5))
# plt.plot(t,ber)
plt.legend()
plt.show()