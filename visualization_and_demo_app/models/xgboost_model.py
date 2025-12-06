import xgboost as xgb
from typing import Optional

MODEL_PARAMS = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'learning_rate': 0.1,
        'max_depth': 12,
        'max_leaves': 16,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42
    }

class XgBoostModel:
    def __init__(self, model: Optional[xgb.XGBRegressor] = None, *args, **kwargs): 
        if model is not None:
            self.model = model
        else:
            self.create_model(*args, **kwargs)

    def create_model(self, *args, **kwargs):
        self.model_params = MODEL_PARAMS
        self.model_params.update(kwargs)
        self.model = xgb.XGBRegressor(**self.model_params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

