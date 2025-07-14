import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

def evaluar_modelo(model, X_test, y_test, verbose=True, graficos=True):
    """
    Evalúa el performance de un modelo de regresión (pipeline completo).
    
    Parámetros:
    - model: pipeline entrenado (incluye preprocesamiento y modelo).
    - X_test: features crudos (sin transformar).
    - y_test: valores reales.
    - verbose: si True, imprime métricas.
    - graficos: si True, muestra visualizaciones.
    
    Devuelve:
    - diccionario con métricas
    """
    # Predicción
    y_pred = model.predict(X_test)

    # Cálculo de métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    resultados = {
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

    if verbose:
        print("🔍 Evaluación del Modelo:")
        for k, v in resultados.items():
            if k == 'MAPE':
                print(f"{k}: {v*100:.2f}%")
            else:
                print(f"{k}: {v:.4f}")

    if graficos:
        residuals = y_test - y_pred

        # 1. Real vs Predicho
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Valor Real')
        plt.ylabel('Valor Predicho')
        plt.title('📈 Real vs Predicho')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2. Histograma de residuos
        plt.figure(figsize=(6,4))
        sns.histplot(residuals, kde=True, bins=30)
        plt.axvline(0, color='red', linestyle='--')
        plt.title('📊 Distribución de residuos')
        plt.xlabel('Error (y_real - y_pred)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 3. Residuos vs Predicciones
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicción')
        plt.ylabel('Residual')
        plt.title('📉 Residuales vs Predicción')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return resultados
