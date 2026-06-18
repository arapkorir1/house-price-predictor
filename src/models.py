from config import RANDOM_SEED, CV_FOLDS, SCORING

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

    model=xgb.XGBRegressor(**defaults)

    return model

def train_model(model,
                X_train,
                y_train,
                X_val=None,
                y_val=None
                ):

    import xgboost as xgb 

    if isinstance(model, xgb.XGBRegressor) and X_val is not None:
        model.fit(
                X_train,
                y_train,
                eval_set=[(X_val,y_val)]
                )
    else:
        model.fit(
                X_train,
                y_train
                )
        
    return model

def cross_validate_model(model,
                         X,
                         y,
                         cv=CV_FOLDS,
                         scoring=SCORING
                         ):
    from sklearn.model_selection import cross_val_score
    import numpy as np

    scores=cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
            )

    results={
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores.tolist()
            }

    return result 

def compare_models(models_dict,
                   X, 
                   y, 
                   cv=CV_FOLDS, 
                   scoring=SCORING):

    import pandas as pd

    results = []

    for name, model in models_dict.items():
        cv_result = cross_validate_model(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        results.append({
            "model": name,
            "mean_score": cv_result["mean"],
            "std_score": cv_result["std"]
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_score", ascending=False)

    return results_df

