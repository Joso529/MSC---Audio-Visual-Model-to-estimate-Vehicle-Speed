from glob import glob
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LearningCurveDisplay, learning_curve
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.pipeline import Pipeline

train_name = pd.read_csv('train_names.csv',header= None).squeeze()
test_name = pd.read_csv('predit_names.csv',header=None)
X_train_pre = 'X_train.csv'
y_train_pre = 'y_train.csv'
X_test_pre = 'X_test.csv'
y_test_pre = 'y_test.csv'
train_classes = 'class.csv'
train_classes = np.loadtxt(train_classes,delimiter=',')

X_train = np.loadtxt(X_train_pre, delimiter=',')
X_test = np.loadtxt(X_test_pre,delimiter=',')
y_train_pre = np.loadtxt(y_train_pre,  delimiter=',')
y_test_pre = np.loadtxt(y_test_pre,delimiter=',')
y_train = y_train_pre
y_test = y_test_pre


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(activation='logistic',solver='sgd',hidden_layer_sizes=(400,200),max_iter=150,
                         learning_rate_init=0.001,batch_size=256,momentum=0.9,alpha=0.00005,
                         n_iter_no_change=10,nesterovs_momentum=True,random_state=45))
])

gkf = GroupKFold(n_splits=len(np.unique(train_classes)))

random_states = list(range(0, 100))
search_space = {
    # 'mlp__activation':['identity','logistic','tanh','relu'],
    # 'mlp__solver':['lbfgs','logistic','tanh','sgd','relu']
    #  'mlp__max_iter': [100,150,200,250,300,350,400,450,500]
    # 'mlp__learning_rate_init': [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]
    #  'mlp__batch_size': [1,2,4,8,16,32,64,128, 256, 512,1024,2048]
    # 'mlp__hidden_layer_sizes': [(400,50),(400,100),(400,150),
    #                              (400,),(400,200),(400,250),(400,300),(400,350),(400,400)]
     #
                # 'mlp__alpha':[(0.00001),(0.00002),(0.00003),(0.00004),(0.00005),(0.0006),(0.00007),(0.00008),(0.00009),(0.0001)]
                # 'mlp__momentum':[(0.1),(0.2),(0.3),(0.4),(0.5),(0.6),(0.7),(0.8),(0.9),(1)]

                # 'mlp__n_iter_no_change':[(1),(2),(5),(10),(15),(20)]  ###no change after 10
                # 'mlp__random_state':random_states
               }

#Grid Search
# mod = GridSearchCV(pipe, search_space, scoring=['r2','neg_root_mean_squared_error'],refit = 'r2',cv = gkf)
# mod.fit(X_train,y_train,groups=train_classes)
# print(mod.best_estimator_)
# print(mod.best_params_)
# print(f'cv:{mod.best_score_}')
# model = mod.best_estimator_
# y_pred = model.predict(X_test)
# mse = root_mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'Root Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')
# # #
# # #
# df = pd.DataFrame(mod.cv_results_)
# df.to_csv("gkf_cv_result.csv")




#LEARNING CURVE
learning = LearningCurveDisplay.from_estimator(pipe, X_train,y_train,
                                              cv=gkf,scoring='r2',groups=train_classes)
plt.title('Learning curve of Audio model')


pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
mse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
# # plt.figure()
# plt.scatter(y_test, y_pred)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('True vs Predicted Values of RMS energy')
plt.show()
#
# da = pd.DataFrame(y_pred)
# da.to_csv("audio_result.csv")

# #cv rmse
# R2cv_score =[]
# RMSEcv_score=[]
# predit_speed = []
# for i, (train_index, test_index) in enumerate(gkf.split(X_train, y_train, groups=train_classes)):
#     print(f"Fold {i + 1}:")
#     pipe = Pipeline([
#         ('scaler', StandardScaler()),
#         ('mlp', MLPRegressor(activation='logistic',solver='sgd',hidden_layer_sizes=(400,200),max_iter=150,
#                          learning_rate_init=0.001,batch_size=256,momentum=0.9,alpha=0.00005,
#                          n_iter_no_change=10,nesterovs_momentum=True,random_state=45))
#     ])
#
#     CVX_train,CVX_test_fold = X_train[train_index], X_train[test_index]
#     CVy_train, CVy_test_fold = y_train[train_index], y_train[test_index]
#
#     train_files = [train_name[idx] for idx in train_index]
#     test_files = [train_name[idx] for idx in test_index]
#     test_vehicle_ids = [train_classes[idx] for idx in test_index]
#
#     # print(f"  Train files: {train_files}")
#     print(f"  Test files: {test_files}")
#     # print(f"  Test vehicle IDs: {test_vehicle_ids}")
#
#     pipe.fit(CVX_train, CVy_train)
#
#     y_pred = pipe.predict(CVX_test_fold)
#     r_score = r2_score(CVy_test_fold, y_pred)
#     mse_score = root_mean_squared_error(CVy_test_fold, y_pred)
#     R2cv_score.append(r_score)
#     RMSEcv_score.append(mse_score)
#     predit_speed.append(y_pred)
#     print(f"  R2 Score for fold {i + 1}: {r_score}")
#     print(f"  RMSE Score for fold {i + 1}: {mse_score}")
#
# my_ndarray = np.array(predit_speed, dtype=object)
# np.save('audio_cv_results.npy', my_ndarray)
# R2_average_score = np.mean(R2cv_score)
# RMSE_average_score = np.mean(RMSEcv_score)
# print(f'Average R2 Score: {R2_average_score}')
# print(f'Average RMSE Score: {RMSE_average_score}')
