from config import RANDOM_SEED

def get_baseline_models():
    '''
    return baseline models
    '''
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression, Ridge 
    from sklearn.ensemble import RandomForestRegressor

    return{
            'dummy':DummyRegressor(strategy='mean'),
            'linear': LinearRegression(),
            'ridge':Ridge(alpha=1.0),
            'randomforest':RandomForestRegressor(
                n_estimators=100,
                n_jobs=1,
                max_depth=10,
                random_state=RANDOM_SEED
                )
            }
'''
regularization is highly sensitive to the scale of  input features; scale for ridge regression
'''

def get_xgboost(task='regression'):
     from sklearn import 
