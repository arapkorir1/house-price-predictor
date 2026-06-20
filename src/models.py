
import joblib
import json
from datetime import datetime
import logging
from config import MODELS_PATH, RANDOM_SEED

logger = logging.getLogger(__name__)


def get_baseline_models(task='regression'):
    """Returns a dict of baseline models to compare."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.dummy import DummyClassifier, DummyRegressor

    if task == 'regression':
        return {
            'dummy':         DummyRegressor(strategy='mean'),
            'ridge':         Ridge(alpha=10),
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        }
    return {
        'dummy':         DummyClassifier(strategy='most_frequent'),
        'logistic':       LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        'random_forest':  RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    }


def train_model(model, X_train, y_train):
    logger.info(f"Training {model.__class__.__name__}...")
    start = datetime.now()
    model.fit(X_train, y_train)
    duration = (datetime.now() - start).seconds
    logger.info(f"Training complete in {duration}s")
    return model


def cross_validate_model(model, X, y, cv=5, scoring='neg_root_mean_squared_error'):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    result = {'mean': scores.mean(), 'std': scores.std(),
              'scores': scores.tolist()}
    logger.info(f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    return result


def compare_models(models_dict, X_train, y_train, cv=5,
                    scoring='neg_root_mean_squared_error'):
    import pandas as pd
    results = []
    for name, model in models_dict.items():
        cv_result = cross_validate_model(model, X_train, y_train,
                                          cv=cv, scoring=scoring)
        results.append({'model': name,
                         'mean_score': cv_result['mean'],
                         'std_score': cv_result['std']})
    return pd.DataFrame(results).sort_values('mean_score', ascending=False)


def save_model(model, name, metadata=None):
    model_path = MODELS_PATH / f'{name}.pkl'
    joblib.dump(model, model_path)
    if metadata:
        meta_path = MODELS_PATH / f'{name}_metadata.json'
        metadata['saved_at']    = datetime.now().isoformat()
        metadata['model_class'] = model.__class__.__name__
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    logger.info(f"Model saved: {model_path}")


def load_model(name):
    path = MODELS_PATH / f'{name}.pkl'
    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model
