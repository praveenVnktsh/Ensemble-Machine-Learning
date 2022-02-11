import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.randomForest import RandomForestClassifier
from ensemble.randomForest import RandomForestRegressor


########### RandomForestClassifier ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

for criteria in ['information_gain', 'gini_index']:
    Classifier_RF = RandomForestClassifier(3, criterion = criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    figs = Classifier_RF.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print()
        print('Class', cls)
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

# # ########### RandomForestRegressor ###################
print('Regressor ---------------')
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

Regressor_RF = RandomForestRegressor(3, criterion = 'variance')
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
