import numpy as np
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
video_results = 'video_results.csv'
video_results = np.loadtxt(video_results, delimiter=',')
audio_results = 'audio_result.csv'
audio_results = np.loadtxt(audio_results, delimiter=',')
predict_labels = 'y_test.csv'
predict_labels = np.loadtxt(predict_labels, delimiter=',')
predict_labels = predict_labels.reshape(-1,1)
cv_labels = np.load('cv_labels.npy', allow_pickle=True)
video_cv_results = np.load('video_cv.npy', allow_pickle=True)
audio_cv_results = np.load('audio_cv_results.npy', allow_pickle=True)
weights = np.arange(0,1.05,0.05)
video_mse = root_mean_squared_error(video_results, predict_labels)
audio_mse = root_mean_squared_error(audio_results, predict_labels)
video_r = r2_score(video_results, predict_labels)
audio_r = r2_score(audio_results, predict_labels)
print(f'video mse:{video_mse},r2:{video_r}')
print(f'audio mse:{audio_mse},r2:{audio_r}')
x = []
mse = []
for i in weights:
    combined_result = audio_results * (1-i) + video_results * (i)
    combined_mse = root_mean_squared_error(combined_result, predict_labels)
    combined_r2 = r2_score(combined_result, predict_labels)
    x.append(i)
    mse.append(combined_mse)
    print(f'Audio:{round(1-i,2)}Video{round(i,2)} Combined mse:{round(combined_mse,3)},r2:{round(combined_r2,3)}')

cv_labels = np.load('cv_labels.npy', allow_pickle=True)
video_cv_results = np.load('video_cv.npy', allow_pickle=True)
audio_cv_results = np.load('audio_cv_results.npy', allow_pickle=True)

com_rmse = []
com_r2 = []
x = []
# for i in weights:
#     temp_rmse = []
#     temp_r2 = []
#     for cv_labels_row, audio_row, video_row in zip(cv_labels, audio_cv_results, video_cv_results):
#         com = audio_row * (1-i) + video_row * (i)
#         rmse = root_mean_squared_error(cv_labels_row, com)
#         r2 = r2_score(cv_labels_row, com)
#         temp_rmse.append(rmse)
#         temp_r2.append(r2)
#     RMSE_average_score = np.mean(temp_rmse)
#     R2_average_score = np.mean(temp_r2)
#     com_rmse.append(RMSE_average_score)
#     com_r2.append(R2_average_score)
#     x.append(i)
#     print(f'Audio:{round(1-i, 2)} Video:{round(i, 2)} Combined mse: {round(RMSE_average_score, 3)}, r2: {round(R2_average_score, 3)}')
#
# plt.scatter(x,com_rmse,marker='x',color='blue',label='cross-validation')
# plt.scatter(x,mse,marker='x',color='orange',label='test')
# plt.legend()
# plt.xlabel('Video Weight (i)\n'
#            'Audio Weight (1-i)')
# plt.ylabel('RMSE (km/h)')
# plt.title('Combined RMSE vs audio,video weights')
# plt.show()
#

