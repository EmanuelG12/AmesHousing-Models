# %% [markdown]
# # <center> EVALUACIÓN

# %%
# Importar librerías
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

sys.path.append('../utils')


from functions import *


# %%
# ----------------------
# 1. Carga del modelo y data
# ----------------------

# Cargar modelo
model = joblib.load('../artifacts/best_model.pkl')

# Cargamos los datos
train = pd.read_csv('../data/processed/train.csv', sep = ',')
test = pd.read_csv('../data/processed/test.csv', sep = ',')

# Separar X e y
X_train = train.drop(columns='SalePrice')
y_train = train['SalePrice']

X_test = test.drop(columns='SalePrice')
y_test = test['SalePrice']

# %%
# ----------------------
# 2. Predicción
# ----------------------
y_pred = model.predict(X_test)
y_pred

# %%
# ----------------------
# 3. Metricas de evaluación
# ----------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R2: {r2:.4f}')

# %%
# ----------------------
# 4. Valor real vs predicho
# ----------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Valor Real')
plt.ylabel('Valor Predicho')
plt.title('Real vs Predicho')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Línea ideal
plt.show()

# %%
# ----------------------
# 5. Valores residuales
# ----------------------
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True)
plt.title('Distribución de Residuales')
plt.xlabel('Residual')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valor Predicho')
plt.ylabel('Residual')
plt.title('Residuals vs Predicho')
plt.show()


