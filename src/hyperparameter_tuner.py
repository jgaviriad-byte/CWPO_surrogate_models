import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
    RBF
)
from sklearn.gaussian_process import GaussianProcessRegressor
df = pd.read_csv("data/processed/Doe_processed.csv")
class CoarseToFineTuner:
    """Clase modular para ejecutar optimización híbrida (Random Search -> Grid Search)

    evitando Data Leakage por lotes/ensayos.
    """

    def __init__(self, estimator, scoring="neg_root_mean_squared_error", cv_splits=5):
        self.estimator = estimator
        self.scoring = scoring
        self.cv_splits = cv_splits
        self.best_params_ = None
        self.best_estimator_ = None

    def fit(self, X, y, groups, param_distributions, grid_search_expansion=0.2, n_iter_random=50, random_state=42):
        """Ejecuta el flujo completo de optimización en dos fases."""
        # Configurar validación cruzada por grupos (agrupado por Ensayo/Lote)
        gkf = GroupKFold(n_splits=self.cv_splits)

        # ----------------------------------------------------
        # FASE 1: Coarse Search con RandomizedSearchCV
        # ----------------------------------------------------
        print("🔍 [FASE 1] Ejecutando RandomizedSearchCV (Exploración Global)...")
        random_search = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=param_distributions,
            n_iter=n_iter_random,
            scoring=self.scoring,
            cv=gkf,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )

        random_search.fit(X, y, groups=groups)
        best_random_params = random_search.best_params_
        print(f"✅ Mejor configuración Fase 1: {best_random_params}\n")

        # ----------------------------------------------------
        # FASE 2: Construcción de Rejilla Fina alrededor del Optimo
        # ----------------------------------------------------
        param_grid = self._build_fine_grid(best_random_params, param_distributions, grid_search_expansion)

        print("🎯 [FASE 2] Ejecutando GridSearchCV (Ajuste Fino)...")
        grid_search = GridSearchCV(
            estimator=self.estimator, param_grid=param_grid, scoring=self.scoring, cv=gkf, n_jobs=-1, refit=True
        )

        grid_search.fit(X, y, groups=groups)

        self.best_params_ = grid_search.best_params_
        self.best_estimator_ = grid_search.best_estimator_

        print(f"🚀 Optimización Completada.")
        print(f"🏆 Mejores Hiperparámetros Finales: {self.best_params_}")
        print(f"📊 Mejor Score ({self.scoring}): {grid_search.best_score_:.4f}")

        return self

    def _build_fine_grid(self, best_params, distributions, expansion_rate):
        """Genera dinámicamente un param_grid reducido centrado en el mejor resultado de la Fase 1."""
        fine_grid = {}

        for param, value in best_params.items():
            if isinstance(value, int):
                # Para enteros (ej. n_estimators, max_depth)
                delta = max(1, int(value * expansion_rate))
                low = max(1, value - delta)
                high = value + delta
                fine_grid[param] = list(np.linspace(low, high, num=3, dtype=int))

            elif isinstance(value, (float, np.floating)):
                # Para continuos (ej. alpha, learning_rate)
                fine_grid[param] = [
                    round(value * (1 - expansion_rate), 4),
                    round(value, 4),
                    round(value * (1 + expansion_rate), 4),
                ]
            else:
                # Para valores categóricos o None
                fine_grid[param] = [value]

        return fine_grid
"""
df = pd.read_csv("data/processed/Doe_processed.csv")
# 1. Cargar tus datos y definir grupos por Ensayo/Reactor
X = df.drop(columns=["C/C0", "Ensayo"])
y = df["C/C0"]
groups = df["Ensayo"]
rf_distributions = {
    "n_estimators": randint(10, 500,),
    "max_depth": [4, 6, 8, 10, 12, None],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", None],
    
}
tuner = CoarseToFineTuner(
estimator=RandomForestRegressor(random_state=42), scoring="neg_root_mean_squared_error", cv_splits=5
)
tuner.fit(
X=X,
y=y,
groups=groups,
param_distributions=rf_distributions,
n_iter_random=1000,  # Prueba 40 combinaciones al azar
)
best_model = tuner.best_estimator_


import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gpr', GaussianProcessRegressor(random_state=42))
])

class GPRHyperparameterTuner:
    """Clase modular para optimizar un Gaussian Process Regressor (GPR)

    evitando la interpolación trivial (R2 = 1.0) mediante el ajuste de ruido
    (alpha/WhiteKernel) y escalas de longitud anisotrópicas.
    """

    def __init__(self, scoring="neg_root_mean_squared_error", cv_splits=5):
        self.scoring = scoring
        self.cv_splits = cv_splits
        self.best_params_ = None
        self.best_estimator_ = None

    def fit(self, X, y, groups, n_iter=30, random_state=42):
        """Ejecuta la búsqueda aleatoria de hiperparámetros para GPR."""
        gkf = GroupKFold(n_splits=self.cv_splits)

        # 1. Definir el número de características para el kernel anisotrópico (ARD)
        n_features = X.shape[1]

        # 2. Espacio de búsqueda de Kernels y Ruido
        # Matern 5/2 con escala por variable + WhiteKernel para tolerancia de ruido experimental
        param_distributions = {
            "alpha": [1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3],  # Nivel de ruido del sensor/lab
            "kernel": [
                Matern(length_scale=[10.0] * n_features, length_scale_bounds=(1e-2, 1e5), nu=2.5)
                for scale in [1.0, 10.0, 50.0, 100.0, 500.0]
            ]
            + [
                RBF(length_scale=[scale] * n_features)
                for scale in [1.0, 10.0, 50.0, 100.0]
            ],
            "n_restarts_optimizer": [0, 2, 5],
        }

        # 3. Inicializar el regresor base
        # Nota: optimizer=None evita colapso de length_scale -> 0
        gpr = GaussianProcessRegressor(
            alpha=0.1,
            optimizer=None,
            random_state=random_state,
        )
        print("🔍 Ejecutando optimización de hiperparámetros para GPR...")
        random_search = RandomizedSearchCV(
            estimator=gpr,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=gkf,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )

        random_search.fit(X, y, groups=groups)

        self.best_params_ = random_search.best_params_
        self.best_estimator_ = random_search.best_estimator_

        print("🚀 Optimización GPR Completada.")
        print(f"🏆 Mejores Hiperparámetros: {self.best_params_}")
        print(f"📊 Mejor Score ({self.scoring}): {random_search.best_score_:.4f}")

        return self

df = pd.read_csv("data/processed/Doe_processed.csv")


# 1. Definir predictores (X), objetivo (y) y lotes (groups)
# ASEGÚRATE de incluir el rezago C/C0(t-1) o el tiempo (min) dentro de X
X = df.drop(columns=["C/C0", "Ensayo"])
y = df["C/C0"]
groups = df["Ensayo"]

# 2. Instanciar y ejecutar el optimizador GPR
gpr_tuner = GPRHyperparameterTuner(
    scoring="neg_root_mean_squared_error", cv_splits=5
)

gpr_tuner.fit(X=X, y=y, groups=groups, n_iter=25)

# 3. Extraer el mejor modelo entrenado
best_gpr_model = gpr_tuner.best_estimator_

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 0. Separar datos en entrenamiento/prueba
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
split_idx = int(len(X) * 0.8)
X_train = X.iloc[indices[:split_idx]]
X_test = X.iloc[indices[split_idx:]]
y_train = y.iloc[indices[:split_idx]]
y_test = y.iloc[indices[split_idx:]]

EPSILON = 1e-5
n_features = X_train.shape[1]

# 1. Kernel Matern con límites flexibles y escala inicial de 1.0
kernel = Matern(
    length_scale=[1.0] * n_features,
    length_scale_bounds=(1e-2, 100.0),
    nu=1.5
)

# 2. Pipeline con optimizador activo
gpr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gpr', GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,
        n_restarts_optimizer=10,
        random_state=42
    ))
])

# 3. TransformedTargetRegressor
model = TransformedTargetRegressor(
    regressor=gpr_pipeline,
    func=lambda y: np.log(np.maximum(y, EPSILON)),
    inverse_func=lambda z: np.exp(z)
)

# 4. Ajustar y Evaluar
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R² Re-evaluado: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:          {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")