import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,GroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LearningCurveDisplay, learning_curve
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.pipeline import Pipeline
X_train_name = pd.read_csv('yolo5x_train_name.csv',header= None).squeeze()
y_test_name = pd.read_csv('yolo5x_train_name.csv')
X_train_pre = 'yolo5x_train_area.csv'
y_train_pre = 'yolo5x_train_label.csv'
X_test_pre = 'yolo5x_test_area.csv'
y_test_pre = 'yolo5x_test_label.csv'
train_classes = 'class.csv'
train_classes = np.loadtxt(train_classes,delimiter=',')

X_train = np.loadtxt(X_train_pre, delimiter=',')
X_test = np.loadtxt(X_test_pre,delimiter=',')
y_train = np.loadtxt(y_train_pre,  delimiter=',')
y_test = np.loadtxt(y_test_pre,delimiter=',')


y_train = y_train.reshape(-1, 1).ravel()
y_test = y_test.reshape(-1,1).ravel()


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(random_state=0))
])
# estimator = MLPRegressor(activation='logistic',solver='lbfgs',hidden_layer_sizes=(100,),learning_rate_init=0.0001,max_iter=200,
#                          alpha=0.002,random_state=4)#(4)
gkf = GroupKFold(n_splits=len(np.unique(train_classes)))

random_states = list(range(0, 100))

search_space = {
    'mlp__activation':['identity','logistic','tanh','relu'],
    'mlp__solver':['lbfgs','logistic','tanh','sgd','relu']
    #  'mlp__max_iter': [100,150,200,250,300,350,400,450,500]
    # 'mlp__learning_rate_init': [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
    #  'mlp__batch_size': [1,2,4,8,16,32,64,128, 256, 512,1024,2048]
    # 'mlp__hidden_layer_sizes': [(250,),(250,),(250, 50),
    #                              (250,100),(250,150),(250,200),(250,250),(250,300),(250,350),(250,450)]
     #
                # 'mlp__alpha':[(0.00001),(0.00002),(0.00003),(0.00004),(0.00005),(0.0006),(0.00007),(0.00008),(0.00009),(0.0001)]
                # 'mlp__momentum':[(0.1),(0.2),(0.3),(0.4),(0.5),(0.6),(0.7),(0.8),(0.9),(1)]

                # 'mlp__n_iter_no_change':[(1),(2),(5),(10),(15),(20)]  ###no change after 10
                # 'mlp__random_state':random_states


}


mod = GridSearchCV(estimator=pipe,param_grid=search_space, scoring=['r2'],refit = 'r2',cv = gkf)
mod.fit(X_train,y_train,groups=train_classes)
print("Best Score: ",mod.best_score_)
print("Best Parameters: ",mod.best_params_)
model = mod.best_estimator_
y_pred = model.predict(X_test)

mse = root_mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)
print(f'Root Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
# # #
# # #
# df = pd.DataFrame(mod.cv_results_)
# df.to_csv("gkf_cv_result.csv")

#
# pipe.fit(X_train,y_train)
# y_pred = pipe.predict(X_test)
# mse = root_mean_squared_error(y_pred, y_test)
# r2 = r2_score(y_pred, y_test)
# print(mse,r2)

# da = pd.DataFrame(y_pred_orig)
# da.to_csv("audio_result.csv")
# plt.figure(figsize=(10,5))
# plt.scatter(y_test, y_pred)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('True vs Predicted Values')
# plt.show()
# #cross validation score
# mse_scores = cross_val_score(estimator,X_train,y_train,cv = gkf,scoring='r2',groups=train_classes)
# print("Scores:", mse_scores)
# print("Mean:",np.mean(mse_scores))

# #LEARNING CURVE
# learning = LearningCurveDisplay.from_estimator(estimator, X_train,y_train,
#                                               cv=gkf,scoring='r2',groups=train_classes)
# plt.title('Learning curve of Video-only model')
# plt.show()



# #save y_pred
# da = pd.DataFrame(y_pred_orig)
# da.to_csv("video_result.csv")

#cv rmse
# R2cv_score =[]
# RMSEcv_score=[]
# predit_speed = []
# for i, (train_index, test_index) in enumerate(gkf.split(X_train, y_train, groups=train_classes)):
#     print(f"Fold {i + 1}:")
#
#     pipe = Pipeline([
#         ('scaler', StandardScaler()),
#         ('mlp',
#          MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(100,), learning_rate_init=0.0001,
#                       max_iter=200,
#                       alpha=0.002, random_state=4))
#     ])
#
#     CVX_train,CVX_test_fold = X_train[train_index], X_train[test_index]
#     CVy_train, CVy_test_fold = y_train[train_index], y_train[test_index]
#
#     train_files = [X_train_name[idx] for idx in train_index]
#     test_files = [X_train_name[idx] for idx in test_index]
#     test_vehicle_ids = [train_classes[idx] for idx in test_index]
#     # print(f"  Train files: {train_files}")
#     print(f"  Test files: {test_files}")
#     # print(f"  Test vehicle IDs: {test_vehicle_ids}")
#
#     # Train the model
#     pipe.fit(CVX_train, CVy_train)
#
#     # Validate the model
#     y_pred = pipe.predict(CVX_test_fold)
#
#     r_score = r2_score(CVy_test_fold, y_pred)
#     mse_score = root_mean_squared_error(CVy_test_fold, y_pred)
#     R2cv_score.append(r_score)
#     RMSEcv_score.append(mse_score)
#     print(f"  R2 Score for fold {i + 1}: {r_score}")
#     print(f"  RMSE Score for fold {i + 1}: {mse_score}")
#
# R2_average_score = np.mean(R2cv_score)
# RMSE_average_score = np.mean(RMSEcv_score)
# print(f'Average R2 Score: {R2_average_score}')
# print(f'Average RMSE Score: {RMSE_average_score}')
