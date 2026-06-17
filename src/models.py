from config import RANDOM_SEED

def get_baseline_model():

    from sklearn.dummy import DummyRegressor
    from sklearn.linear_models import LInearRegressor,Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    return{
            'dummy': DummyRegressor(strategy='mean'),
            'linear': LinearRegression(),
            'ridge': RidgeRegressor(alpha=1),
            'randomforest': RandomForestFRegressor(
                n_estimators=500,
                max_depth=6,
                n_jobs=1, 
                random_state=RANDOM_SEED
                )
            }

def get_xgboost(**kwargs):

    import xgboost as xgb

    defaults= {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_curve': 0.05,
            'random_state': RANDOM_SEED
            'n-jobs': 1
            }
    defaults.update(kwargs)

    return xgb.XGBRegressor(**defaults)

def 


