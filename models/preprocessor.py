# %% [markdown]
# # <center> PREPROCESAMIENTO DE LOS DATOS

# %%
# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings

df = pd.read_csv('../data/raw/AmesHousing.csv', sep=',')

# %% [markdown]
# Gracias al análisis exploratorio de datos (EDA), descubrimos que nuestro conjunto de datos contiene muchas columnas que no aportan un valor significativo al modelo. Por lo tanto, antes de llevar a cabo el preprocesamiento, decidimos conservar solo las 5 columnas más importantes

# %%
# ----------------------
# 1. SelectKBest 
# ----------------------

X = df.drop(columns =['SalePrice', 'Garage Area'])       # Eliminamos columna objetivo
X = pd.get_dummies(X, drop_first=True)                  # Pasamos a dummies variables catégoricas
X = X.fillna(0)                                         # Remplazamos nulos por 0

y = df['SalePrice']                                     # Variable objetivo

selector = SelectKBest(score_func=f_regression, k=5)    # Seleccionamos mejores
selector.fit(X, y)  # Entrenamos

# %%
# Obtener las 10 mejores columnas
selected_columns = X.columns[selector.get_support()]
print("Top 5 columnas seleccionadas:\n", selected_columns)

df = X
df = df[selected_columns]
df['SalePrice'] = y  # Asignamos la variable y a la columna 'SalePrice'

# %% [markdown]
# ##### Significado columnas:
# - Overall Qual: Calidad general de la casa, evaluada por el evaluador. Valor de 1 (muy mala) a 10 (excelente). Es una de las variables más correlacionadas con SalePrice. <br>
# - Total Bsmt SF: Superficie total del sótano (basement) en pies cuadrados.<br>
# - 1st Flr SF:	Superficie del primer piso en pies cuadrados.<br>
# - Gr Liv Area: Área habitable sobre el suelo (Ground Living Area), en pies cuadrados. No incluye sótano. Muy correlacionada con el precio.<br>
# - Garage Cars: Capacidad de la cochera en cantidad de autos.<br>
# - SalePrice: Precio final de venta de la casa. Es la variable objetivo (target) para predecir.<br>

# %% [markdown]
# ### Modificamos distribuciones para un mejor desempeño

# %%
# ----------------------
# 2. Distribuciones
# ----------------------
# Versiones transformadas
df['Total Bsmt SF'] = np.sqrt(df['Total Bsmt SF'])
df['1st Flr SF'] = np.log1p(df['1st Flr SF'])
df['Gr Liv Area'] = np.log1p(df['Gr Liv Area'])
df['SalePrice'] = np.log1p(df['SalePrice'])

# %%
# ----------------------
# 3. Escalamos
# ----------------------

# Definimos las columnas numéricas que queremos escalar
cols_to_scale = ['Total Bsmt SF', '1st Flr SF', 'Gr Liv Area']

# Creamos un ColumnTransformer para escalar solo esas columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), cols_to_scale)
    ],
    remainder='passthrough'  # deja pasar las demás columnas sin tocar
)

# Separar X e y
X = df.drop(columns='SalePrice')
y = df['SalePrice']

# Ajustamos el preprocesador y transformamos los datos
X_scaled = preprocessor.fit_transform(X)

# Convertimos a DataFrame con los nombres originales
X_scaled = pd.DataFrame(X_scaled, columns=cols_to_scale + [col for col in X.columns if col not in cols_to_scale])

# Guardamos el preprocesador ya entrenado
joblib.dump(preprocessor, '../artifacts/preprocessor.pkl')

# %%
# ----------------------
# 4. Split
# ----------------------

# Variable dependiente
y = df['SalePrice']

# Split
X_train,X_test,y_train,y_test= train_test_split(X_scaled, y, test_size=0.30, random_state=42)

# Unimos para guardarlos como .csv
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# %%
# ----------------------
# 5. Save
# ----------------------
# Guardar en carpeta processed
train.to_csv('../data/processed/train.csv', index=False)
test.to_csv('../data/processed/test.csv', index=False)


