import lightgbm as lgb
from typing import Optional

MODEL_PARAMS = {
        'objective':'regression',
        'metric':'mae',
        'n_estimators':500,
        'learning_rate':0.1,
        'max_depth':12,
        'num_leaves':16,
        'min_child_samples':10,
        'reg_alpha':0,
        'reg_lambda':1,
        'random_state':42
    }

class LightGBMModel:
    def __init__(self, model: Optional[lgb.LGBMRegressor] = None, *args, **kwargs): 
        if model is not None:
            self.model = model
        else:
            self.create_model(*args, **kwargs)

    def create_model(self, *args, **kwargs):
        self.model_params = MODEL_PARAMS
        self.model_params.update(kwargs)
        self.model = lgb.LGBMRegressor(**self.model_params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

