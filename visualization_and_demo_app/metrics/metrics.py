from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def eval_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    r2 = r2_score(y_test, y_pred)
    print(f"R squared score: {r2:.4f}")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

def feature_importance_from_model_plot(model, X):
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })

    feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances from Model')
    plt.tight_layout()
    plt.show()