# %% [markdown]
# # <center> MODELADO

# %%
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Cargar datasets procesados
train = pd.read_csv('../data/processed/train.csv')
test = pd.read_csv('../data/processed/test.csv')

# Separar X e y
X_train = train.drop(columns='SalePrice')
y_train = train['SalePrice']

X_test = test.drop(columns='SalePrice')
y_test = test['SalePrice']

# %%
# Cargar preprocesador entrenado
preprocessor = joblib.load('../artifacts/preprocessor.pkl')

# Modelos
models = {
    'lr': LinearRegression(),
    'dt': DecisionTreeRegressor(),
    'knr': KNeighborsRegressor(),
    'rf': RandomForestRegressor()
}

# Param Grid con claves correctas (todas las claves deben empezar con 'model__')
param_grid = [
    {
        'model': [models['lr']],
        'model__fit_intercept': [True, False],
    },
    {
        'model': [models['dt']],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    {
        'model': [models['knr']],
        'model__n_neighbors': [3, 5, 7],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    },
    {
        'model': [models['rf']],
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }
]

# Pipeline correcto
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())  # Este se sobrescribe por GridSearchCV
])

# Grid Search con CV
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

# Entrenamiento
grid.fit(X_train, y_train)

# Resultados
print("Mejor modelo:", grid.best_estimator_)
print("Mejores hiperparámetros:", grid.best_params_)
print("Mejor score (R²):", grid.best_score_)

# %%
# Guardamos el preprocesador ya entrenado
joblib.dump(grid.best_estimator_, '../artifacts/best_model.pkl')


