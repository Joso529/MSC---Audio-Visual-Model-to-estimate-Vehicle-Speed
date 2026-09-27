import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

cv_labels = np.load('predit_labels.npy', allow_pickle=True)
video_cv_results = np.load('predit_speed.npy', allow_pickle=True)
audio_cv_results = np.load('audio_cv_results.npy', allow_pickle=True)
train_vehicle = pd.read_csv('train_vehicle.csv',header= None).squeeze()
a = []

weights = np.arange(0,1.05,0.05)
for cv_labels_row, audio_row, video_row in zip(cv_labels, audio_cv_results, video_cv_results):
    com = audio_row * 0.2 + video_row * 0.8
    rmse = root_mean_squared_error(cv_labels_row, com)
    a.append(rmse)
print(a)

x = []
mse = []
row = 6
for i in weights:
    combined_result = audio_cv_results[row] * (1-i) + video_cv_results[row] * (i)
    rmse = root_mean_squared_error(cv_labels[row], combined_result)
    x.append(i)
    mse.append(rmse)
    print(f'Audio:{round(1-i,2)}Video{round(i,2)} Combined mse:{round(rmse,3)}')
plt.figure(figsize=(8,6))
plt.scatter(x,mse,marker='x',color='blue')
plt.xlabel('Video Weight (i)\n'
           'Audio Weight (1-i)')
plt.ylabel('RMSE (km/h)')
plt.title(f'Combined RMSE vs audio,video weights of {train_vehicle[row]}')
plt.show()
print(train_vehicle[row])