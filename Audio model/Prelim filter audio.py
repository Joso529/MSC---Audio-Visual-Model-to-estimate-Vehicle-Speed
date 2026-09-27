import os
import librosa
from glob import glob
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import scipy.signal
import scipy.io.wavfile
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.pipeline import Pipeline
split_file = glob('C:/Users/jonat/Downloads/carnoises/**/Train_valid_split.txt')
train_audio = []
train_labels = []
train_names = []
train_pass_by = []
train_type = []
FRAME_SIZE = 2048
HOP_LENGTH = 1024
predit_audio = []
predit_labels = []
predit_names = []
predit_passby = []
predit_type = []

for i in range(len(split_file)):
    f = open(split_file[i],'r')
    train_split = f.readlines()

    for j in range(len(train_split)):
        file_list = train_split[j]
        file,test_valid = file_list.split(' ', 1)
        test_valid = test_valid.strip()
        audio_files = glob(f'C:/Users/jonat/Downloads/carnoises/**/{file}.wav')
        annotations = glob(f'C:/Users/jonat/Downloads/carnoises/**/{file}.txt')
        txt = open(annotations[0], 'r')
        vehicle = file.split('_')[0]
        separate = txt.read().split(' ')
        speed = float(separate[0])
        pass_by = float(separate[1])
        txt.close()
        for audio in audio_files:
            y, sr = librosa.load(audio_files[0], sr=44100)
            start = int((pass_by - 1) * sr)
            end = int((pass_by + 1) * sr)
            sos = scipy.signal.butter(2, 300, 'highpass', fs=sr, output='sos')
            filtered_data = scipy.signal.sosfiltfilt(sos, y)
            filtered_y = filtered_data[start:end]
            lb = librosa.feature.rms(y=filtered_y,frame_length = FRAME_SIZE,hop_length=HOP_LENGTH)[0]
            if test_valid == 'train':
                train_audio.append(lb)
                train_labels.append(speed)
                train_pass_by.append(pass_by)
                train_names.append(file)
                train_type.append(vehicle)
                f.close()
            else:
                predit_audio.append(lb)
                predit_labels.append(speed)
                predit_passby.append(pass_by)
                predit_names.append(file)
                predit_type.append(vehicle)
                f.close()
if (len(predit_audio)+len(train_audio)) == 400:
    print('data loaded succesfully')


train_audio = np.array(train_audio)
predit_audio = np.array(predit_audio)
X_train = (train_audio)
y_train = np.array(train_labels)

X_test =(predit_audio)
y_test = np.array(predit_labels)
train_classes = 'class.csv'
train_classes = np.loadtxt(train_classes,delimiter=',')

#
# #
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(activation='logistic',solver='sgd',random_state=0))
])

gkf = GroupKFold(n_splits=len(np.unique(train_classes)))
random_states = list(range(0, 100))
search_space = {
    'mlp__activation':['identity','logistic','tanh','relu'],
    'mlp__solver':['lbfgs','logistic','tanh','sgd','relu']

               }

# #Grid Search
# mod = GridSearchCV(pipe, search_space, scoring=['r2'],refit = 'r2',cv = gkf)
# mod.fit(X_train,y_train,groups=train_classes)
# print(mod.best_estimator_)
# print(mod.best_params_)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
mse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


#cv rmse
R2cv_score =[]
RMSEcv_score=[]
for i, (train_index, test_index) in enumerate(gkf.split(X_train, y_train, groups=train_classes)):
    # print(f"Fold {i + 1}:")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(activation='logistic',solver='sgd',random_state=0))  # 83:8.12
    ])

    CVX_train,CVX_test_fold = X_train[train_index], X_train[test_index]
    CVy_train, CVy_test_fold = y_train[train_index], y_train[test_index]

    train_files = [train_names[idx] for idx in train_index]
    test_files = [train_names[idx] for idx in test_index]
    # print(f"  Test files: {test_files}")
    pipe.fit(CVX_train, CVy_train)
    y_pred = pipe.predict(CVX_test_fold)
    r_score = r2_score(CVy_test_fold, y_pred)
    mse_score = root_mean_squared_error(CVy_test_fold, y_pred)
    R2cv_score.append(r_score)
    RMSEcv_score.append(mse_score)
    print(f"  R2 Score {i + 1}: {r_score}")
    print(f"  RMSE Score {i + 1}: {mse_score}")

R2_average_score = np.mean(R2cv_score)
RMSE_average_score = np.mean(RMSEcv_score)
print(f'Average R2 Score: {R2_average_score}')
print(f'Average RMSE Score: {RMSE_average_score}')
