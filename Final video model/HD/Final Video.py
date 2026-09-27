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
    ('mlp', MLPRegressor(activation='relu',solver='lbfgs',hidden_layer_sizes=(300,),max_iter=150,
                          learning_rate_init=0.00005,alpha=0.00008,random_state=0))
])

gkf = GroupKFold(n_splits=len(np.unique(train_classes)))

random_states = list(range(0, 100))

search_space = {
    'mlp__activation':['identity','logistic','tanh','relu'],
    'mlp__solver':['lbfgs','logistic','tanh','sgd','relu']
    #  'mlp__max_iter': [100,150,200,250,300,350,400,450,500]
    # 'mlp__learning_rate_init': [0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001]
    # 'mlp__hidden_layer_sizes': [(300,),(300,50),(300,100),
    #                              (300,150),(300,200),(300,250),(300,300)]
     #
                # 'mlp__alpha':[(0.00001),(0.00002),(0.00003),(0.00004),(0.00005),(0.0006),(0.00007),(0.00008),(0.00009),(0.0001)]

                # 'mlp__random_state':random_states


}


# mod = GridSearchCV(estimator=pipe,param_grid=search_space, scoring=['r2'],refit = 'r2',cv = gkf)
# mod.fit(X_train,y_train,groups=train_classes)
# print("Score: ",mod.best_score_)
# print("Parameters: ",mod.best_params_)

#
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
mse = root_mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)
print(mse,r2)

# da = pd.DataFrame(y_pred)
# da.to_csv("video_results.csv")


# Plot the learning curve
# learning = LearningCurveDisplay.from_estimator(pipe, X_train, y_train, cv=gkf, scoring='r2', groups=train_classes)
# plt.title('Learning curve of Video model')
# plt.show()

# da = pd.DataFrame(y_pred)
# da.to_csv("video_result.csv")

# #cv rmse
R2cv_score =[]
RMSEcv_score=[]
predit_speed = []
predit_labels=[]
for i, (train_index, test_index) in enumerate(gkf.split(X_train, y_train, groups=train_classes)):
    print(f"Fold {i + 1}:")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp',
         MLPRegressor(activation='relu',solver='lbfgs',hidden_layer_sizes=(300,),max_iter=150,
                         learning_rate_init=0.00005,alpha=0.00008,random_state=0))
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
    predit_labels.append(CVy_test_fold)
    predit_speed.append(y_pred)
    print(f"  R2 Score for fold {i + 1}: {r_score}")
    print(f"  RMSE Score for fold {i + 1}: {mse_score}")

R2_average_score = np.mean(R2cv_score)
RMSE_average_score = np.mean(RMSEcv_score)

my_ndarray = np.array(predit_labels, dtype=object)
np.save('predit_labels.npy', my_ndarray)

da = pd.DataFrame(predit_labels)
da.to_csv("cv_labels.csv")
db = pd.DataFrame(predit_speed)
db.to_csv("cv_speed.csv")
print(f'Average R2 Score: {R2_average_score}')
print(f'Average RMSE Score: {RMSE_average_score}')
