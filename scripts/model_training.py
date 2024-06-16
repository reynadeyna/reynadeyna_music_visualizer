import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import os

model_path = 'models/'

if not os.path.exists(model_path):
    os.makedirs(model_path)

df = pd.read_csv('data/audio_visual_dataset_augmented.csv')

import ast
X = np.vstack(df['features'].apply(lambda x: np.array(ast.literal_eval(x))))
y = np.vstack(df['visual_params'].apply(lambda x: np.array(ast.literal_eval(x))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for i in range(y_train.shape[1]):
    print(f'Training model for visual parameter {i}')
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1]
    }
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train[:, i])
    
    model_filename = f'{model_path}model_visual_param_{i}.pkl'
    joblib.dump(grid_search.best_estimator_, model_filename)
    
    y_pred = grid_search.predict(X_test)
    mse = np.mean((y_pred - y_test[:, i]) ** 2)
    print(f'Best parameters for visual parameter {i}: {grid_search.best_params_}')
    print(f'Mean Squared Error for visual parameter {i}: {mse}')
    print(f'Model saved to {model_filename}')

print('Training complete.')
