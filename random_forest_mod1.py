from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from data.readingDataFromDb.trainTestSplitMarketSplit import X_train, X_test, y_train, y_test


def rfr_model(X, y):
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],                               
                                random_state=False, verbose=False)
# Perform K-Fold CV
    scores = cross_val_predict(rfr, X, y, cv=10)
    return scores, rfr
    

scores, rfr = rfr_model(X_train, np.ravel(y_train,order='C'))
predictions = cross_val_predict(rfr, X_test, np.ravel(y_test,order='C'), cv=10)
MAE_testSet = mean_absolute_error(y_test , predictions)
print("MAE of the model is ", MAE_testSet)
