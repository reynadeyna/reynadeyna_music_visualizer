import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

model_path = 'models/'

if not os.path.exists(model_path):
    os.makedirs(model_path)

df = pd.read_csv('data/audio_visual_dataset_augmented.csv')

X = np.vstack(df['features'].apply(lambda x: np.array(eval(x))))
y = np.vstack(df['visual_params'].apply(lambda x: np.array(eval(x))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'SVR': Pipeline([('scaler', StandardScaler()), ('svr', SVR())]),
    'MLPRegressor': MLPRegressor(random_state=42),
    'XGBRegressor': xgb.XGBRegressor(random_state=42)
}

param_grids = {
    'DecisionTreeRegressor': {'max_depth': [3, 4, 5]},
    'RandomForestRegressor': {'n_estimators': [50, 100], 'max_depth': [3, 4, 5]},
    'GradientBoostingRegressor': {'n_estimators': [50, 100], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1]},
    'SVR': {'svr__C': [1, 10], 'svr__gamma': ['scale', 'auto']},
    'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
    'XGBRegressor': {'n_estimators': [50, 100], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1]}
}

results = []

for model_name, model in models.items():
    for i in range(y_train.shape[1]):
        print(f"Training {model_name} for visual parameter {i}")
        
        if model_name in param_grids:
            grid_search = GridSearchCV(model, param_grids[model_name], cv=2, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train[:, i])
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model.fit(X_train, y_train[:, i])
            best_model = model
            best_params = None
        
        joblib.dump(best_model, f'{model_path}{model_name}_model_visual_param_{i}.pkl')
        
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test[:, i], y_pred)
        
        results.append({
            'Model': model_name,
            'Visual Parameter': i,
            'Best Params': best_params,
            'Mean Squared Error': mse
        })

results_df = pd.DataFrame(results)

results_df.to_csv('model_comparison_results.csv', index=False)

print(results_df)

plt.figure(figsize=(10, 6))
for model_name in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model_name]
    plt.plot(subset['Visual Parameter'], subset['Mean Squared Error'], label=model_name)

plt.xlabel('Visual Parameter')
plt.ylabel('Mean Squared Error')
plt.title('Model Comparison')
plt.legend()
plt.show()
