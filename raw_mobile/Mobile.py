import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold,GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
X_train_name = pd.read_csv('mb_train_name.csv',header= None).squeeze()
y_test_name = pd.read_csv('mb_train_name.csv')
X_train_pre = 'mb_train_area.csv'
y_train_pre = 'mb_train_label.csv'
X_test_pre = 'mb_test_area.csv'
y_test_pre = 'mb_test_label.csv'
train_classes = 'C:/Users/jonat/OneDrive/Desktop/detect/detect/class.csv'
train_classes = np.loadtxt(train_classes,delimiter=',')

X_train = np.loadtxt(X_train_pre, delimiter=',')
X_test = np.loadtxt(X_test_pre,delimiter=',')
y_train = np.loadtxt(y_train_pre,  delimiter=',')
y_test = np.loadtxt(y_test_pre,delimiter=',')


y_train = y_train.reshape(-1, 1).ravel()
y_test = y_test.reshape(-1,1).ravel()


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

#
# mod = GridSearchCV(estimator=pipe,param_grid=search_space, scoring=['r2'],refit = 'r2',cv = gkf)
# mod.fit(X_train,y_train,groups=train_classes)
# print("Score:",mod.best_score_)
# print("Parameters:",mod.best_params_)
#
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
mse = root_mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)
print(mse,r2)

#cv rmse
R2cv_score =[]
RMSEcv_score=[]
predit_speed = []
for i, (train_index, test_index) in enumerate(gkf.split(X_train, y_train, groups=train_classes)):
    print(f"Fold {i + 1}:")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp',
         MLPRegressor(activation='logistic',solver='sgd',random_state=0))
    ])

    CVX_train,CVX_test_fold = X_train[train_index], X_train[test_index]
    CVy_train, CVy_test_fold = y_train[train_index], y_train[test_index]

    train_files = [X_train_name[idx] for idx in train_index]
    test_files = [X_train_name[idx] for idx in test_index]
    test_vehicle_ids = [train_classes[idx] for idx in test_index]
    # print(f"  Train files: {train_files}")
    print(f"  Test files: {test_files}")
    # print(f"  Test vehicle IDs: {test_vehicle_ids}")
    pipe.fit(CVX_train, CVy_train)
    y_pred = pipe.predict(CVX_test_fold)
    r_score = r2_score(CVy_test_fold, y_pred)
    mse_score = root_mean_squared_error(CVy_test_fold, y_pred)
    R2cv_score.append(r_score)
    RMSEcv_score.append(mse_score)
    print(f"  R2 Score for fold {i + 1}: {r_score}")
    print(f"  RMSE Score for fold {i + 1}: {mse_score}")

R2_average_score = np.mean(R2cv_score)
RMSE_average_score = np.mean(RMSEcv_score)
print(f'Average R2 Score: {R2_average_score}')
print(f'Average RMSE Score: {RMSE_average_score}')
