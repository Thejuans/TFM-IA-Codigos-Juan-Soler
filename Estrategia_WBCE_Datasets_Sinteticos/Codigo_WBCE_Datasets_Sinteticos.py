import json
import time
import warnings

import numpy as np
import pandas as pd

# Backend no interactivo para evitar errores de Tkinter al guardar figuras cuando se usa paralelización con joblib en Windows/VS Code.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from pathlib import Path
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from itertools import product

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import ConvergenceWarning

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, roc_auc_score)
from imblearn.metrics import (sensitivity_score, specificity_score, geometric_mean_score, make_index_balanced_accuracy)

# ============================================================
# 0. CONFIGURACIÓN GENERAL
# ============================================================

SEEDS = list(range(1, 21))

# Rejilla completa: len(C_VALUES) x len(C_VALUES) combinaciones de pesos por semilla. (pesos c1 y c0)
C_VALUES = np.round(np.logspace(np.log10(0.01), np.log10(10), 30), 6).tolist() # 30 valores log-espaciados entre 0.01 y 10, redondeados a 6 decimales.

N_JOBS = -1  # -1 = usamos todos los núcleos de CPU disponibles para paralelizar por semilla

# Parámetro técnico de optimización. No se estudia como hiperparámetro; se fija
# para evitar avisos de no convergencia del solver lbfgs cuando la rejilla usa
# pesos muy extremos. El hiperparámetro experimental sigue siendo solo (c0, c1).
MODEL_MAX_ITER = 500


# Modelo de referencia sin ponderación diferencial entre clases.
# Equivale al punto (c0=1, c1=1) de la rejilla manual.
EQUAL_WEIGHTS_C0 = 1.0
EQUAL_WEIGHTS_C1 = 1.0
FINAL_MODEL_NAMES = {"best_val_f1_grid", "sklearn_balanced", "equal_weights"}

# Paleta común para mapas 2D, superficies 3D, leyendas y comparativas finales.
MODEL_COLORS = {
    "best_val_f1_grid": "#D55E00",   # naranja/vermillion suave
    "sklearn_balanced": "#0072B2",   # azul profundo
    "equal_weights": "#CC79A7",      # violeta suave
}

PIN_EDGE_COLOR = "black"

PIN_TEXT_COLOR = "black"

# Etiquetas legibles y homogéneas para las figuras de la memoria.
# No cambia nombres de columnas ni archivos: solo modifica títulos, ejes,
# leyendas y barras de color para evitar guiones bajos en las imágenes.
SPLIT_DISPLAY_NAMES = {
    "train": "TRAIN",
    "val": "VALIDACIÓN",
    "test": "TEST",
}

STAT_DISPLAY_NAMES = {
    "mean": "Mean",
    "std": "STD",
}

METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "balanced_accuracy": "Balanced Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "fpr": "FPR",
    "fnr": "FNR",
    "gmean": "G-Mean",
    "iba": "IBA",
    "mcc": "MCC",
    "kappa": "Kappa",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
}


def format_scenario_label_for_plot(scenario_name):
    """
    Nombre legible del escenario en títulos de figuras.
    Evita guiones bajos y mantiene un formato uniforme para la memoria.
    """
    return str(scenario_name).replace("_", " ").title()


def split_metric_name(metric):
    """
    Separa una métrica interna del código en:
    - split: val/test/train si aparece;
    - métrica base: f1, balanced_accuracy, pr_auc, roc_auc, etc.;
    - estadístico: mean/std si aparece.

    Ejemplo:
    val_balanced_accuracy_mean -> (val, balanced_accuracy, mean)
    """
    metric_text = str(metric)
    split_name = None
    stat_name = None

    for prefix in ["train_", "val_", "test_"]:
        if metric_text.startswith(prefix):
            split_name = prefix[:-1]
            metric_text = metric_text[len(prefix):]
            break

    for suffix in ["_mean", "_std"]:
        if metric_text.endswith(suffix):
            stat_name = suffix[1:]
            metric_text = metric_text[:-len(suffix)]
            break

    return split_name, metric_text, stat_name


def format_metric_base_label(metric_base):
    """
    Convierte el nombre interno de una métrica en una etiqueta legible.
    """
    metric_base = str(metric_base)

    return METRIC_DISPLAY_NAMES.get(
        metric_base,
        metric_base.replace("_", " ").title(),
    )


def format_metric_label_for_plot(metric, include_split=False, include_stat=True):
    """
    Etiqueta completa de métrica para títulos, leyendas y barras de color.
    """
    split_name, metric_base, stat_name = split_metric_name(metric)

    parts = []

    if include_split and split_name is not None:
        parts.append(SPLIT_DISPLAY_NAMES.get(split_name, split_name.upper()))

    parts.append(format_metric_base_label(metric_base))

    if include_stat and stat_name is not None:
        parts.append(STAT_DISPLAY_NAMES.get(stat_name, stat_name.upper()))

    return " · ".join(parts)


def format_metric_axis_label_for_plot(metric):
    """
    Etiqueta corta para ejes Y. Se omite split y mean/std para no recargar.
    """
    _, metric_base, _ = split_metric_name(metric)
    return format_metric_base_label(metric_base)

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "outputs_datasets_sinteticos"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATASETS_DIR = OUTPUT_DIR / "datasets"
COEFFICIENTS_WIDE_DIR = OUTPUT_DIR / "coefficients_wide"
GRID_MANUAL_COEFFICIENTS_WIDE_DIR = COEFFICIENTS_WIDE_DIR / "grid_manual"
EQUAL_WEIGHTS_COEFFICIENTS_WIDE_DIR = COEFFICIENTS_WIDE_DIR / "equal_weights"
SKLEARN_BALANCED_COEFFICIENTS_WIDE_DIR = COEFFICIENTS_WIDE_DIR / "sklearn_balanced"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
COEFFICIENTS_WIDE_DIR.mkdir(parents=True, exist_ok=True)
GRID_MANUAL_COEFFICIENTS_WIDE_DIR.mkdir(parents=True, exist_ok=True)
EQUAL_WEIGHTS_COEFFICIENTS_WIDE_DIR.mkdir(parents=True, exist_ok=True)
SKLEARN_BALANCED_COEFFICIENTS_WIDE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 0.1. ORGANIZACIÓN DE FIGURAS
# ============================================================

def ensure_directory(path):
    """
    Crea una carpeta si no existe y devuelve la ruta.
    Solo afecta a la organización de salidas; no cambia el entrenamiento ni la evaluación.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_scenario_figure_path(scenario_name, filename):
    """
    Devuelve la ruta organizada de una figura propia de un escenario.

    Estructura final simplificada:
    outputs_datasets_sinteticos/figures/<escenario>/<archivo.png>

    """
    scenario_dir = FIGURES_DIR / str(scenario_name)
    ensure_directory(scenario_dir)
    return scenario_dir / filename


def get_global_figure_path(filename):
    """
    Devuelve la ruta organizada de una figura global comparativa.
    """
    global_dir = FIGURES_DIR / "comparativas_globales"
    ensure_directory(global_dir)
    return global_dir / filename


# Outputs principales de la rejilla de VALIDACIÓN.
RAW_RESULTS_PATH = OUTPUT_DIR / "results_raw_multiseed.csv"
RAW_PREDICTIONS_PATH = OUTPUT_DIR / "predictions_raw_multiseed.csv"  # Predicciones de VALIDACIÓN para todos los puntos de la rejilla.
AGG_RESULTS_PATH = OUTPUT_DIR / "results_aggregated_mean_std.csv"
BEST_CONFIGS_PATH = OUTPUT_DIR / "best_configs_by_val_f1_mean.csv"
SCENARIOS_CONFIG_PATH = OUTPUT_DIR / "scenarios_config.csv"
TIMERS_PATH = OUTPUT_DIR / "timers_execution.txt"
EXPERIMENT_METADATA_PATH = OUTPUT_DIR / "experiment_metadata.json"
RUN_SIGNATURE = "synthetic_logreg_standard_full_grid_v8_grid30_train_val_test_sin_metrica_descartada"

# Punto equivalente a class_weight='balanced' de Scikit-Learn.
BALANCED_POINTS_PATH = OUTPUT_DIR / "balanced_sklearn_points_by_scenario.csv"
BALANCED_POINTS_RAW_PATH = OUTPUT_DIR / "balanced_sklearn_points_raw_by_seed.csv"

# Comparación final en VALIDACIÓN: best_val_f1_grid vs sklearn_balanced vs equal_weights.
VAL_FINAL_RAW_RESULTS_PATH = OUTPUT_DIR / "val_final_results_raw_multiseed.csv"
VAL_FINAL_RAW_PREDICTIONS_PATH = OUTPUT_DIR / "val_final_predictions_selected_models_raw_multiseed.csv"
VAL_FINAL_AGG_RESULTS_PATH = OUTPUT_DIR / "val_final_results_aggregated_mean_std.csv"
VAL_FINAL_COMPARISON_PATH = OUTPUT_DIR / "val_final_comparison_table.csv"

# Comparación final en TEST: best_val_f1_grid vs sklearn_balanced vs equal_weights.
TEST_FINAL_RAW_RESULTS_PATH = OUTPUT_DIR / "test_final_results_raw_multiseed.csv"
TEST_FINAL_RAW_PREDICTIONS_PATH = OUTPUT_DIR / "test_final_predictions_raw_multiseed.csv"
TEST_FINAL_AGG_RESULTS_PATH = OUTPUT_DIR / "test_final_results_aggregated_mean_std.csv"
TEST_FINAL_COMPARISON_PATH = OUTPUT_DIR / "test_final_comparison_table.csv"

print("Los resultados se guardarán en:")
print(OUTPUT_DIR.resolve())

iba_metric = make_index_balanced_accuracy(alpha=0.1, squared=True)(geometric_mean_score)

# ============================================================
# 1. ESCENARIOS
# ============================================================

# En este trabajo se interpreta la clase 1 como la clase positiva/caso de interés y la clase 0 como la clase negativa/no caso. En un contexto sanitario, la clase 1 puede representar presencia de condición, riesgo o enfermedad.

SCENARIOS = {
    "ideal": {
        "n_samples": 1000, # Número de muestras o filas totales del dataset (incluye train, val y test).
        "n_features": 20, # Número total de características o columnas de entrada (sin contar la columna target ni split). 
        "n_informative": 5,
        "n_redundant": 5, 
        "n_repeated": 0,
        "weights": [0.50, 0.50],
        "class_sep": 1.8,
        "flip_y": 0.0, # No se introduce ruido aleatorio en la variable objetivo y; las etiquetas de clase no se modifican artificialmente.
        "n_clusters_per_class": 1, # Para que sea más fácil de aprender, cada clase tiene un solo cluster. Esto hace que la frontera de decisión sea más clara y el modelo pueda capturar la relación entre características y clases sin tener que lidiar con múltiples subgrupos dentro de cada clase.
    },
    "intermedio": {
        "n_samples": 1000,
        "n_features": 100,
        "n_informative": 5,
        "n_redundant": 10,
        "n_repeated": 10,
        "weights": [0.70, 0.30],
        "class_sep": 1.0,
        "flip_y": 0.0,
        "n_clusters_per_class": 1,
    },
    "avanzado": {
        "n_samples": 1000,
        "n_features": 180,
        "n_informative": 5,
        "n_redundant": 12,
        "n_repeated": 15,
        "weights": [0.85, 0.15],
        "class_sep": 0.75,
        "flip_y": 0.0,
        "n_clusters_per_class": 1,
    },
    "dificil": {
        "n_samples": 1000,
        "n_features": 300,
        "n_informative": 5,
        "n_redundant": 15,
        "n_repeated": 20,
        "weights": [0.95, 0.05],
        "class_sep": 0.5,
        "flip_y": 0.0,
        "n_clusters_per_class": 1,
    },
}

scenario_rows = []
for scenario_name, config in SCENARIOS.items():
    row = {"scenario": scenario_name}
    row.update(config)
    scenario_rows.append(row)

pd.DataFrame(scenario_rows).to_csv(SCENARIOS_CONFIG_PATH, index=False)

# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================


def load_or_create_dataset(seed, scenario_name, config):
    scenario_dataset_dir = DATASETS_DIR / scenario_name
    scenario_dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = scenario_dataset_dir / f"{scenario_name}_seed_{seed:02d}_dataset.csv"

    expected_feature_cols = [f"x_{i:03d}" for i in range(config["n_features"])]
    required_cols = set(expected_feature_cols + ["target", "split", "sample_id"])

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        existing_cols = set(df.columns)

        dataset_is_compatible = required_cols.issubset(existing_cols)

        if dataset_is_compatible:
            return df

        print(f"Dataset antiguo/incompatible detectado y regenerado: {dataset_path}")
        dataset_path.unlink()

    feature_names = expected_feature_cols

    X, y = make_classification(
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        n_informative=config["n_informative"],
        n_redundant=config["n_redundant"],
        n_repeated=config["n_repeated"],
        n_classes=2,
        n_clusters_per_class=config["n_clusters_per_class"],
        weights=config["weights"],
        class_sep=config["class_sep"],
        flip_y=config["flip_y"],
        random_state=seed,
        shuffle=False,
    )

    # Split: 72% train, 18% val, 10% test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=seed, stratify=y)  # Stratify para mantener la proporción de clases en cada split, aunque con pocas muestras de clase 1 en algunos escenarios, puede que no se mantenga exactamente.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=seed, stratify=y_train)

    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train["target"] = y_train
    df_train["split"] = "train"

    df_val = pd.DataFrame(X_val, columns=feature_names)
    df_val["target"] = y_val
    df_val["split"] = "val"

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test["target"] = y_test
    df_test["split"] = "test"

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df["sample_id"] = np.arange(len(df))

    df.to_csv(dataset_path, index=False)

    return df


def compute_metrics_for_split(y_true, y_pred, y_prob, split_name):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    if len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        pr_auc = np.nan
        roc_auc = np.nan

    metrics = {
        f"{split_name}_accuracy": accuracy_score(y_true, y_pred),
        f"{split_name}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{split_name}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{split_name}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{split_name}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{split_name}_sensitivity": sensitivity_score(y_true, y_pred),
        f"{split_name}_specificity": specificity_score(y_true, y_pred),
        f"{split_name}_fpr": fpr,
        f"{split_name}_fnr": fnr,
        f"{split_name}_gmean": geometric_mean_score(y_true, y_pred),
        f"{split_name}_iba": iba_metric(y_true, y_pred),
        f"{split_name}_mcc": matthews_corrcoef(y_true, y_pred),
        f"{split_name}_kappa": cohen_kappa_score(y_true, y_pred),
        f"{split_name}_pr_auc": pr_auc,
        f"{split_name}_roc_auc": roc_auc,
    }

    return metrics


def compute_sklearn_balanced_weights(y_train):
    """
    Calcula los pesos equivalentes a class_weight='balanced' usando directamente la función oficial compute_class_weight de Scikit-Learn.
    """
    y_train = np.asarray(y_train, dtype=int)
    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    c0_balanced = float(weights[0])
    c1_balanced = float(weights[1])

    return c0_balanced, c1_balanced


def compute_balanced_points_by_scenario():
    """
    Calcula el punto balanced de Scikit-Learn para cada escenario.
    Como con flip_y=0.0 y particiones estratificadas los conteos de clase en train son constantes, estos pesos coinciden entre seeds y se usan como punto Sklearn balanced en las figuras de validación mean.
    """
    rows = []

    for scenario_name, config in SCENARIOS.items():
        for seed in SEEDS:
            df = load_or_create_dataset(seed, scenario_name, config)
            train_df = df[df["split"] == "train"].reset_index(drop=True)
            y_train = train_df["target"].to_numpy(dtype=int)

            c0_balanced, c1_balanced = compute_sklearn_balanced_weights(y_train)

            rows.append({
                "scenario": scenario_name,
                "seed": seed,
                "balanced_c0": c0_balanced,
                "balanced_c1": c1_balanced,
            })

    balanced_raw_df = pd.DataFrame(rows)
    balanced_raw_df.to_csv(BALANCED_POINTS_RAW_PATH, index=False)

    balanced_df = (
        balanced_raw_df
        .groupby("scenario")[["balanced_c0", "balanced_c1"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    balanced_df.columns = [
        "scenario",
        "balanced_c0_mean",
        "balanced_c0_std",
        "balanced_c1_mean",
        "balanced_c1_std",
    ]

    balanced_df.to_csv(BALANCED_POINTS_PATH, index=False)

    return balanced_df


# ============================================================
# 3. FUNCIÓN PARA UNA SEMILLA: REJILLA EN TRAIN/VALIDACIÓN
# ============================================================

def make_logistic_model(class_weight, seed):
    """
    Regresión logística estándar de Scikit-Learn.

    Se mantiene fija la configuración base del modelo. El único elemento que
    cambia en el experimento es class_weight, es decir, los pesos c0 y c1.

    Nota sobre random_state:
    con el solver por defecto (lbfgs), la regresión logística es prácticamente
    determinista y random_state apenas influye en el ajuste. Se deja por
    coherencia con el uso de semillas del estudio,
    pero no se considera un hiperparámetro experimental.
    """
    return LogisticRegression(
        random_state=seed,
        class_weight=class_weight,
        max_iter=MODEL_MAX_ITER,
    )


def fit_model_with_convergence_info(model, X_train_scaled, y_train):
    """
    Entrena el modelo capturando si aparece ConvergenceWarning.

    Esto no cambia el entrenamiento; solo registra información diagnóstica:
    - número de iteraciones usadas por el solver,
    - max_iter del modelo,
    - si apareció aviso de convergencia,
    - si alcanzó el máximo de iteraciones.
    """
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train_scaled, y_train)

    convergence_warning_messages = [
        str(w.message)
        for w in caught_warnings
        if issubclass(w.category, ConvergenceWarning)
    ]

    convergence_warning = len(convergence_warning_messages) > 0
    n_iter = int(np.max(model.n_iter_)) if hasattr(model, "n_iter_") else np.nan
    max_iter = int(model.max_iter) if hasattr(model, "max_iter") else np.nan
    reached_max_iter = bool(n_iter >= max_iter) if not pd.isna(n_iter) and not pd.isna(max_iter) else False

    convergence_info = {
        "n_iter": n_iter,
        "max_iter": max_iter,
        "convergence_warning": convergence_warning,
        "reached_max_iter": reached_max_iter,
        "converged_without_warning": not convergence_warning,
        "convergence_message": " | ".join(convergence_warning_messages),
    }

    return convergence_info


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} min"
    hours = minutes / 60
    return f"{hours:.2f} h"


def get_grid_models_per_seed():
    """
    Número de modelos entrenados por seed en la rejilla manual.
    """
    return len(C_VALUES) * len(C_VALUES)


def get_total_grid_train_models():
    """
    Número total de modelos entrenados en TRAIN por la rejilla manual.
    """
    return len(SEEDS) * get_grid_models_per_seed()


def get_total_single_strategy_train_models():
    """
    Número total de modelos entrenados en TRAIN por estrategias con un único modelo por seed.
    Aplica a Equal weights y Sklearn balanced.
    """
    return len(SEEDS)



def run_pandas_numpy_metrics_warmup():
    """
    Warm-up técnico de pandas, NumPy/BLAS y métricas.

    Inicializa operaciones usadas después en agregación, selección, predicción
    y cálculo de métricas. Este tiempo no se suma al coste comparable.
    """
    t_start = time.perf_counter()

    try:
        rng = np.random.default_rng(12345)
        grid_size = len(C_VALUES) * len(C_VALUES)
        n_warmup_seeds = min(len(SEEDS), 20)
        n_rows = grid_size * n_warmup_seeds

        c0_grid, c1_grid = np.meshgrid(
            np.asarray(C_VALUES, dtype=float),
            np.asarray(C_VALUES, dtype=float),
            indexing="ij",
        )

        warmup_df = pd.DataFrame({
            "scenario": np.repeat("warmup", n_rows),
            "seed": np.tile(np.arange(1, n_warmup_seeds + 1), grid_size),
            "c0": np.repeat(c0_grid.ravel(), n_warmup_seeds),
            "c1": np.repeat(c1_grid.ravel(), n_warmup_seeds),
            "val_f1": rng.random(n_rows),
            "val_pr_auc": rng.random(n_rows),
            "val_balanced_accuracy": rng.random(n_rows),
            "val_mcc": rng.normal(0.0, 0.2, n_rows),
        })

        warmup_agg = (
            warmup_df
            .groupby(["scenario", "c0", "c1"])[["val_f1", "val_pr_auc", "val_balanced_accuracy", "val_mcc"]]
            .agg(["mean", "std"])
            .reset_index()
        )

        _ = warmup_df.sort_values(["scenario", "seed", "c0", "c1"]).reset_index(drop=True)
        _ = warmup_agg["val_f1"].idxmax()
        _ = pd.concat([warmup_df.head(20), warmup_df.tail(20)], ignore_index=True)

        X = rng.normal(size=(400, 40))
        betas = rng.normal(size=40)
        logits = X @ betas
        probs = sigmoid_stable(logits)
        preds = (probs >= 0.5).astype(int)
        y_true = rng.integers(0, 2, size=400)

        _ = np.mean(X, axis=0)
        _ = np.std(X, axis=0)
        _ = np.bincount(y_true.astype(int), minlength=2)
        _ = compute_metrics_for_split(y_true, preds, probs, "warmup")

        elapsed = time.perf_counter() - t_start
        print(f"Warm-up pandas/NumPy/métricas terminado: {format_seconds(elapsed)}. No se incluye en el coste comparable.")
    except Exception as exc:
        print(f"Warm-up pandas/NumPy/métricas omitido por error no crítico: {exc}")
        print("El script continúa, pero la primera agregación/selección podría incluir algo de sobrecoste inicial.")


def _joblib_warmup_task(index):
    """
    Warm-up técnico realista y acotado.

    Inicializa joblib, Scikit-Learn, StandardScaler, LogisticRegression y BLAS
    antes de medir los bloques reales. Este tiempo no se suma al coste comparable.
    """
    rng = np.random.default_rng(1000 + int(index))
    X = rng.normal(size=(240, 12))
    y = np.array([0] * 192 + [1] * 48, dtype=int)
    order = rng.permutation(len(y))
    X = X[order]
    y = y[order]

    X_scaled = StandardScaler().fit_transform(X)

    warmup_class_weights = [
        {0: 1.0, 1: 1.0},
        "balanced",
        {0: 0.5, 1: 2.0},
    ]

    for class_weight in warmup_class_weights:
        model = LogisticRegression(
            random_state=int(index) + 1,
            class_weight=class_weight,
            max_iter=min(50, MODEL_MAX_ITER),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_scaled, y)
            _ = model.predict_proba(X_scaled[:20])[:, 1]

    return index


def preload_datasets_for_timing():
    """
    Precarga los datasets/splits antes de medir.

    Evita que la primera estrategia medida cargue en solitario el coste de
    lectura/generación de datasets o caché del sistema operativo. No se incluye
    en el coste comparable porque es calentamiento técnico común.
    """
    t_start = time.perf_counter()
    n_loaded = 0

    try:
        for scenario_name, config in SCENARIOS.items():
            for seed in SEEDS:
                df = load_or_create_dataset(seed, scenario_name, config)
                feature_cols = [c for c in df.columns if c.startswith("x_")]
                train_df = df[df["split"] == "train"].reset_index(drop=True)
                if not train_df.empty and feature_cols:
                    X_train = train_df[feature_cols].to_numpy(dtype=float)
                    y_train = train_df["target"].to_numpy(dtype=int)
                    _ = StandardScaler().fit_transform(X_train)
                    _ = np.bincount(y_train.astype(int), minlength=2)
                n_loaded += 1

        elapsed = time.perf_counter() - t_start
        print(f"Precarga técnica de datasets terminada: {n_loaded} splits/semillas · {format_seconds(elapsed)}. No se incluye en el coste comparable.")
    except Exception as exc:
        print(f"Precarga técnica de datasets omitida por error no crítico: {exc}")
        print("El script continúa, pero el primer bloque medido podría incluir algo de sobrecoste inicial.")


def run_parallel_warmup():
    """
    Warm-up técnico previo a los bloques medidos.

    Precarga datasets e inicializa pandas, NumPy/BLAS, métricas, joblib y
    Scikit-Learn. Este tiempo no se guarda como coste comparable de ninguna
    estrategia.
    """
    print("\nWarm-up técnico: precargando datasets e inicializando pandas, NumPy, métricas, joblib y sklearn.")
    preload_datasets_for_timing()
    run_pandas_numpy_metrics_warmup()

    n_tasks = max(1, min(len(SEEDS), 20))
    t_start = time.perf_counter()
    try:
        Parallel(n_jobs=N_JOBS)(delayed(_joblib_warmup_task)(i) for i in range(n_tasks))
        elapsed = time.perf_counter() - t_start
        print(f"Warm-up joblib/sklearn terminado: {format_seconds(elapsed)}. No se incluye en el coste comparable.")
    except Exception as exc:
        print(f"Warm-up joblib/sklearn omitido por error no crítico: {exc}")
        print("El script continúa, pero el primer bloque medido podría incluir algo de sobrecoste inicial.")


def sigmoid_stable(logits):
    """Sigmoid con recorte para evitar overflow numérico."""
    logits = np.clip(np.asarray(logits, dtype=float), -709, 709)
    return 1.0 / (1.0 + np.exp(-logits))


def predict_from_saved_coefficients(X_scaled, intercept, betas):
    """
    Predice probabilidades y clases usando coeficientes ya guardados.

    Esto evita volver a entrenar el modelo final manual cuando ya existen los
    coeficientes de la rejilla para ese punto (c0, c1).
    """
    logits = X_scaled @ np.asarray(betas, dtype=float) + float(intercept)
    y_prob = sigmoid_stable(logits)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_prob, y_pred


def get_coefficients_wide_path(scenario_name, strategy_name):
    """
    Ruta del archivo wide de coeficientes para cada estrategia.

    Se separan las estrategias para que:
    - la rejilla manual quede como superficie completa de pesos,
    - equal_weights exista como estrategia independiente aunque (1,1) se quite de la rejilla,
    - sklearn_balanced guarde sus pesos y coeficientes propios por semilla.
    """
    if strategy_name == "grid_manual":
        return GRID_MANUAL_COEFFICIENTS_WIDE_DIR / f"{scenario_name}_grid_manual_coefficients_wide.csv"

    if strategy_name == "equal_weights":
        return EQUAL_WEIGHTS_COEFFICIENTS_WIDE_DIR / f"{scenario_name}_equal_weights_coefficients_wide.csv"

    if strategy_name == "sklearn_balanced":
        return SKLEARN_BALANCED_COEFFICIENTS_WIDE_DIR / f"{scenario_name}_sklearn_balanced_coefficients_wide.csv"

    raise ValueError(f"Estrategia de coeficientes no reconocida: {strategy_name}")


def extract_convergence_info_from_row(row):
    """
    Recupera información de convergencia almacenada junto a los coeficientes.
    """
    return {
        "n_iter": row.get("n_iter", np.nan),
        "max_iter": row.get("max_iter", np.nan),
        "convergence_warning": bool(row.get("convergence_warning", False)),
        "reached_max_iter": bool(row.get("reached_max_iter", False)),
        "converged_without_warning": row.get("converged_without_warning", np.nan),
        "convergence_message": row.get("convergence_message", ""),
    }


def get_saved_coefficients(seed_coeffs_df, c0, c1, feature_cols, require_weight_match=True):
    """
    Recupera coeficientes guardados en formato wide.

    Para la rejilla manual se exige que coincidan c0 y c1.
    Para estrategias finales guardadas de forma independiente también se puede
    exigir la coincidencia de pesos, pero ya no dependen de que el punto exista
    dentro de C_VALUES.
    """
    if seed_coeffs_df is None or seed_coeffs_df.empty:
        return None

    missing_feature_cols = [col for col in feature_cols if col not in seed_coeffs_df.columns]
    if missing_feature_cols:
        return None

    if require_weight_match:
        if "c0" not in seed_coeffs_df.columns or "c1" not in seed_coeffs_df.columns:
            return None

        mask = (
            np.isclose(seed_coeffs_df["c0"].astype(float).to_numpy(), float(c0))
            & np.isclose(seed_coeffs_df["c1"].astype(float).to_numpy(), float(c1))
        )
        rows = seed_coeffs_df.loc[mask]
    else:
        rows = seed_coeffs_df

    if rows.empty:
        return None

    row = rows.iloc[0]
    intercept = float(row["intercept"])
    betas = row[feature_cols].to_numpy(dtype=float)

    return {
        "intercept": intercept,
        "betas": betas,
        "convergence_info": extract_convergence_info_from_row(row),
    }


def load_seed_coefficients(scenario_name, seed, strategy_name):
    """
    Carga los coeficientes guardados de una estrategia, escenario y semilla.
    """
    coeff_path = get_coefficients_wide_path(scenario_name, strategy_name)

    if not coeff_path.exists():
        return None

    coeffs_df = pd.read_csv(coeff_path)
    coeffs_df = coeffs_df[coeffs_df["seed"] == seed].reset_index(drop=True)

    if coeffs_df.empty:
        return None

    return coeffs_df


def add_prediction_rows(predictions_list, scenario_name, seed, model_name, split_name, split_df, y_true, y_pred, y_prob, c0, c1):
    """
    Añade predicciones/probabilidades a una lista de resultados.
    """
    for sample_id, y_true_i, y_pred_i, y_prob_i in zip(
        split_df["sample_id"],
        y_true,
        y_pred,
        y_prob,
    ):
        predictions_list.append({
            "scenario": scenario_name,
            "seed": seed,
            "model_name": model_name,
            "split": split_name,
            "sample_id": int(sample_id),
            "c0": float(c0),
            "c1": float(c1),
            "y_true": int(y_true_i),
            "y_pred": int(y_pred_i),
            "y_prob": float(y_prob_i),
        })


def process_one_seed_grid_manual_train(seed, scenario_name, config):
    """
    FASE TRAIN · Rejilla manual para una semilla.

    En esta fase solo se entrena con TRAIN y se guardan los coeficientes de
    todos los puntos (c0, c1). No se evalúa todavía en validación ni en test.
    """
    t_seed_start = time.perf_counter()

    df = load_or_create_dataset(seed, scenario_name, config)
    feature_cols = [c for c in df.columns if c.startswith("x_")]

    train_df = df[df["split"] == "train"].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["target"].to_numpy(dtype=int)

    t_scaler_start = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaler_seconds = time.perf_counter() - t_scaler_start

    seed_coefficients = []

    fit_seconds_total = 0.0
    convergence_warnings_total = 0
    reached_max_iter_total = 0

    for c0 in C_VALUES:
        for c1 in C_VALUES:
            class_weights = {0: c0, 1: c1}

            model = make_logistic_model(
                class_weight=class_weights,
                seed=seed,
            )

            t_fit_start = time.perf_counter()
            convergence_info = fit_model_with_convergence_info(model, X_train_scaled, y_train)
            fit_seconds_total += time.perf_counter() - t_fit_start

            convergence_warnings_total += int(convergence_info["convergence_warning"])
            reached_max_iter_total += int(convergence_info["reached_max_iter"])

            row_coefficients = {
                "scenario": scenario_name,
                "seed": seed,
                "model_name": "grid_manual",
                "c0": c0,
                "c1": c1,
                "intercept": float(model.intercept_[0]),
                "n_iter": convergence_info["n_iter"],
                "max_iter": convergence_info["max_iter"],
                "convergence_warning": convergence_info["convergence_warning"],
                "reached_max_iter": convergence_info["reached_max_iter"],
                "converged_without_warning": convergence_info["converged_without_warning"],
                "convergence_message": convergence_info["convergence_message"],
            }

            for feature_name, beta in zip(feature_cols, model.coef_[0]):
                row_coefficients[feature_name] = float(beta)

            seed_coefficients.append(row_coefficients)

    total_seconds = time.perf_counter() - t_seed_start

    timer_row = {
        "scenario": scenario_name,
        "seed": seed,
        "stage": "grid_manual_train",
        "model_name": "grid_manual",
        "n_models": len(C_VALUES) * len(C_VALUES),
        "scaler_seconds": scaler_seconds,
        "fit_seconds": fit_seconds_total,
        "validation_prediction_seconds": 0.0,
        "test_prediction_seconds": 0.0,
        "metrics_seconds": 0.0,
        "convergence_warnings": convergence_warnings_total,
        "reached_max_iter_count": reached_max_iter_total,
        "total_seconds": total_seconds,
        "total_time_readable": format_seconds(total_seconds),
        "used_saved_coefficients": False,
        "skipped_because_existing_outputs": False,
    }

    return pd.DataFrame(seed_coefficients), pd.DataFrame([timer_row])


def process_one_seed_grid_manual_validation(seed, scenario_name, config):
    """
    FASE VALIDACIÓN · Rejilla manual para una semilla.

    En esta fase no se entrena. Se cargan los coeficientes de la rejilla manual
    entrenados previamente con TRAIN, se predice sobre VALIDACIÓN y se calculan
    las métricas de validación para cada punto (c0, c1).
    """
    t_seed_start = time.perf_counter()

    df = load_or_create_dataset(seed, scenario_name, config)
    feature_cols = [c for c in df.columns if c.startswith("x_")]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["target"].to_numpy(dtype=int)

    t_scaler_start = time.perf_counter()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_val_scaled = scaler.transform(X_val)
    scaler_seconds = time.perf_counter() - t_scaler_start

    seed_coeffs_df = load_seed_coefficients(
        scenario_name=scenario_name,
        seed=seed,
        strategy_name="grid_manual",
    )

    if seed_coeffs_df is None or seed_coeffs_df.empty:
        raise FileNotFoundError(
            f"No se encontraron coeficientes de la rejilla manual para "
            f"scenario={scenario_name}, seed={seed}. Ejecuta antes la fase TRAIN."
        )

    seed_results = []
    seed_predictions = []

    val_prediction_seconds_total = 0.0
    metrics_seconds_total = 0.0
    convergence_warnings_total = 0
    reached_max_iter_total = 0

    for c0 in C_VALUES:
        for c1 in C_VALUES:
            saved = get_saved_coefficients(
                seed_coeffs_df=seed_coeffs_df,
                c0=c0,
                c1=c1,
                feature_cols=feature_cols,
                require_weight_match=True,
            )

            if saved is None:
                raise FileNotFoundError(
                    f"Faltan coeficientes guardados para scenario={scenario_name}, "
                    f"seed={seed}, c0={c0}, c1={c1}."
                )

            convergence_info = saved["convergence_info"]
            convergence_warnings_total += int(convergence_info.get("convergence_warning", False))
            reached_max_iter_total += int(convergence_info.get("reached_max_iter", False))

            t_pred_start = time.perf_counter()
            y_val_prob, y_val_pred = predict_from_saved_coefficients(
                X_val_scaled,
                intercept=saved["intercept"],
                betas=saved["betas"],
            )
            val_prediction_seconds_total += time.perf_counter() - t_pred_start

            add_prediction_rows(
                seed_predictions,
                scenario_name=scenario_name,
                seed=seed,
                model_name="grid_manual",
                split_name="val",
                split_df=val_df,
                y_true=y_val,
                y_pred=y_val_pred,
                y_prob=y_val_prob,
                c0=c0,
                c1=c1,
            )

            row_result = {
                "scenario": scenario_name,
                "seed": seed,
                "c0": c0,
                "c1": c1,
                "n_iter": convergence_info.get("n_iter", np.nan),
                "max_iter": convergence_info.get("max_iter", np.nan),
                "convergence_warning": convergence_info.get("convergence_warning", False),
                "reached_max_iter": convergence_info.get("reached_max_iter", False),
                "converged_without_warning": convergence_info.get("converged_without_warning", np.nan),
                "convergence_message": convergence_info.get("convergence_message", ""),
            }

            t_metrics_start = time.perf_counter()
            row_result.update(
                compute_metrics_for_split(y_val, y_val_pred, y_val_prob, "val")
            )
            metrics_seconds_total += time.perf_counter() - t_metrics_start

            seed_results.append(row_result)

    total_seconds = time.perf_counter() - t_seed_start

    timer_row = {
        "scenario": scenario_name,
        "seed": seed,
        "stage": "grid_manual_validation",
        "model_name": "grid_manual",
        "n_models": len(C_VALUES) * len(C_VALUES),
        "scaler_seconds": scaler_seconds,
        "fit_seconds": 0.0,
        "validation_prediction_seconds": val_prediction_seconds_total,
        "test_prediction_seconds": 0.0,
        "metrics_seconds": metrics_seconds_total,
        "convergence_warnings": convergence_warnings_total,
        "reached_max_iter_count": reached_max_iter_total,
        "total_seconds": total_seconds,
        "total_time_readable": format_seconds(total_seconds),
        "used_saved_coefficients": True,
        "skipped_because_existing_outputs": False,
    }

    return (
        pd.DataFrame(seed_results),
        pd.DataFrame(seed_predictions),
        pd.DataFrame([timer_row]),
    )


# ============================================================
# ============================================================
# 4. COEFICIENTES DE ESTRATEGIAS FINALES INDEPENDIENTES
# ============================================================

def make_wide_coefficient_row(scenario_name, seed, model_name, c0, c1, intercept, betas, feature_cols, convergence_info):
    """
    Construye una fila wide de coeficientes: metadatos + intercept + un beta por variable.
    """
    row = {
        "scenario": scenario_name,
        "seed": seed,
        "model_name": model_name,
        "c0": float(c0),
        "c1": float(c1),
        "intercept": float(intercept),
        "n_iter": convergence_info["n_iter"],
        "max_iter": convergence_info["max_iter"],
        "convergence_warning": convergence_info["convergence_warning"],
        "reached_max_iter": convergence_info["reached_max_iter"],
        "converged_without_warning": convergence_info["converged_without_warning"],
        "convergence_message": convergence_info["convergence_message"],
    }

    for feature_name, beta in zip(feature_cols, betas):
        row[feature_name] = float(beta)

    return row



def process_one_seed_single_final_strategy_coefficients(seed, scenario_name, config, strategy_name):
    """
    FASE TRAIN · Estrategia final independiente para una semilla.

    Entrena y guarda coeficientes de una única estrategia:
    - equal_weights: c0=1, c1=1.
    - sklearn_balanced: pesos calculados automáticamente desde y_train.

    Esta separación permite medir tiempos reales independientes para cada
    estrategia, manteniendo la paralelización por semillas.
    """
    t_seed_start = time.perf_counter()

    df = load_or_create_dataset(seed, scenario_name, config)
    feature_cols = [c for c in df.columns if c.startswith("x_")]

    train_df = df[df["split"] == "train"].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["target"].to_numpy(dtype=int)

    t_scaler_start = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaler_seconds = time.perf_counter() - t_scaler_start

    balanced_c0, balanced_c1 = compute_sklearn_balanced_weights(y_train)

    if strategy_name == "equal_weights":
        strategy = {
            "model_name": "equal_weights",
            "class_weight": {0: EQUAL_WEIGHTS_C0, 1: EQUAL_WEIGHTS_C1},
            "c0": EQUAL_WEIGHTS_C0,
            "c1": EQUAL_WEIGHTS_C1,
        }
    elif strategy_name == "sklearn_balanced":
        strategy = {
            "model_name": "sklearn_balanced",
            "class_weight": "balanced",
            "c0": balanced_c0,
            "c1": balanced_c1,
        }
    else:
        raise ValueError(f"Estrategia final no reconocida: {strategy_name}")

    model = make_logistic_model(
        class_weight=strategy["class_weight"],
        seed=seed,
    )

    t_fit_start = time.perf_counter()
    convergence_info = fit_model_with_convergence_info(model, X_train_scaled, y_train)
    fit_seconds = time.perf_counter() - t_fit_start

    coefficient_row = make_wide_coefficient_row(
        scenario_name=scenario_name,
        seed=seed,
        model_name=strategy["model_name"],
        c0=strategy["c0"],
        c1=strategy["c1"],
        intercept=float(model.intercept_[0]),
        betas=model.coef_[0],
        feature_cols=feature_cols,
        convergence_info=convergence_info,
    )

    total_seconds = time.perf_counter() - t_seed_start

    timer_row = {
        "scenario": scenario_name,
        "seed": seed,
        "stage": "final_strategy_coefficients_train",
        "model_name": strategy["model_name"],
        "n_models": 1,
        "scaler_seconds": scaler_seconds,
        "fit_seconds": fit_seconds,
        "validation_prediction_seconds": 0.0,
        "test_prediction_seconds": 0.0,
        "metrics_seconds": 0.0,
        "convergence_warnings": int(convergence_info["convergence_warning"]),
        "reached_max_iter_count": int(convergence_info["reached_max_iter"]),
        "n_iter": convergence_info["n_iter"],
        "max_iter": convergence_info["max_iter"],
        "total_seconds": total_seconds,
        "total_time_readable": format_seconds(total_seconds),
        "used_saved_coefficients": False,
        "skipped_because_existing_outputs": False,
    }

    return pd.DataFrame([coefficient_row]), pd.DataFrame([timer_row])



def final_strategy_coefficients_are_compatible():
    """
    Comprueba que existen los coeficientes wide independientes de equal_weights y sklearn_balanced.
    """
    required_cols = {"scenario", "seed", "model_name", "c0", "c1", "intercept", "n_iter", "max_iter"}
    expected_seeds = set(SEEDS)

    for scenario_name in SCENARIOS.keys():
        for strategy_name in ["equal_weights", "sklearn_balanced"]:
            coeff_path = get_coefficients_wide_path(scenario_name, strategy_name)

            if not coeff_path.exists():
                return False

            try:
                df = pd.read_csv(coeff_path)
            except Exception:
                return False

            if not required_cols.issubset(set(df.columns)):
                return False

            if set(df["scenario"].unique().tolist()) != {scenario_name}:
                return False

            if set(df["seed"].astype(int).unique().tolist()) != expected_seeds:
                return False

            if set(df["model_name"].unique().tolist()) != {strategy_name}:
                return False

            if set(df["max_iter"].astype(int).unique().tolist()) != {MODEL_MAX_ITER}:
                return False

            if strategy_name == "equal_weights":
                if not np.allclose(df["c0"].astype(float).to_numpy(), EQUAL_WEIGHTS_C0):
                    return False
                if not np.allclose(df["c1"].astype(float).to_numpy(), EQUAL_WEIGHTS_C1):
                    return False

            if len(df) != len(SEEDS):
                return False

    return True


def train_or_load_final_strategy_coefficients(timer_frames):
    """
    Genera los coeficientes de equal_weights y sklearn_balanced si no existen.

    En esta versión se entrenan y se miden como bloques independientes:
    - TRAIN equal_weights.
    - TRAIN sklearn_balanced.

    Cada bloque sigue paralelizando por semillas con joblib, pero se obtiene un
    tiempo real separado para cada estrategia.
    """
    if final_strategy_coefficients_are_compatible():
        t_load_start = time.perf_counter()
        print("\nYa existen coeficientes wide compatibles para equal_weights y sklearn_balanced.")
        print("Se reutilizan y no se reentrenan esas estrategias finales.\n")
        add_wall_clock_timer(
            timer_frames,
            scenario_name="all",
            stage="load_existing_final_strategy_coefficients",
            start_time=t_load_start,
            n_models=0,
            skipped=True,
        )
        return

    print("\nSe generan coeficientes wide independientes para equal_weights y sklearn_balanced.")
    print("Cada estrategia se entrena en un bloque paralelo separado para medir tiempos reales independientes.\n")

    for scenario_name, config in SCENARIOS.items():
        for strategy_name in ["equal_weights", "sklearn_balanced"]:
            scenario_wall_start = time.perf_counter()

            print("\n" + "=" * 80)
            print(f"TRAIN · {strategy_name.upper()} · ESCENARIO: {scenario_name.upper()}")
            print("=" * 80)
            print(f"Ejecutando {len(SEEDS)} semillas en paralelo...")
            print("Modelos entrenados: 1 por seed")

            parallel_output = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_seed_single_final_strategy_coefficients)(
                    seed,
                    scenario_name,
                    config,
                    strategy_name,
                )
                for seed in SEEDS
            )

            scenario_coeffs = [item[0] for item in parallel_output]
            scenario_timers = [item[1] for item in parallel_output]

            scenario_coeffs_df = pd.concat(scenario_coeffs, ignore_index=True)
            scenario_timers_df = pd.concat(scenario_timers, ignore_index=True)
            timer_frames.append(scenario_timers_df)

            n_strategy_models = int(scenario_timers_df["n_models"].fillna(0).sum()) if not scenario_timers_df.empty else 0
            n_strategy_warnings = int(scenario_timers_df["convergence_warnings"].fillna(0).sum()) if "convergence_warnings" in scenario_timers_df.columns and not scenario_timers_df.empty else 0
            n_strategy_max_iter = int(scenario_timers_df["reached_max_iter_count"].fillna(0).sum()) if "reached_max_iter_count" in scenario_timers_df.columns and not scenario_timers_df.empty else 0
            print(
                f"Convergencia TRAIN {strategy_name} {scenario_name}: "
                f"warnings={n_strategy_warnings}/{n_strategy_models}, "
                f"alcanzan max_iter={n_strategy_max_iter}/{n_strategy_models}"
            )

            strategy_df = (
                scenario_coeffs_df[scenario_coeffs_df["model_name"] == strategy_name]
                .sort_values(["scenario", "seed", "model_name"])
                .reset_index(drop=True)
            )

            strategy_df.to_csv(
                get_coefficients_wide_path(scenario_name, strategy_name),
                index=False,
            )

            add_wall_clock_timer(
                timer_frames,
                scenario_name=scenario_name,
                stage=f"{strategy_name}_train_scenario_parallel_total",
                start_time=scenario_wall_start,
                n_models=len(SEEDS),
                skipped=False,
            )

            print(f"Coeficientes guardados para {strategy_name} en escenario {scenario_name}.")

# ============================================================
# 5. EVALUACIÓN FINAL EN VALIDACIÓN Y TEST
# ============================================================

def evaluate_one_final_model(
    seed,
    scenario_name,
    split_name,
    model_name,
    class_weight,
    c0,
    c1,
    X_train_scaled,
    y_train,
    X_eval_scaled,
    y_eval,
    eval_df,
    feature_cols,
    seed_coeffs_df=None,
    use_saved_coefficients_if_possible=False,
    require_weight_match=True,
):
    """
    Evalúa un modelo final en validación o test.

    Para los modelos finales se intenta usar primero el archivo de coeficientes
    correspondiente a cada estrategia. Así se evita reentrenar en la evaluación
    final cuando ya existen coeficientes guardados. Si faltan, se detiene la ejecución para no mezclar entrenamiento con validación/test.
    """
    t_total_start = time.perf_counter()
    fit_seconds = 0.0
    prediction_seconds = 0.0
    metrics_seconds = 0.0
    used_saved_coefficients = False
    convergence_info = {
        "n_iter": np.nan,
        "max_iter": np.nan,
        "convergence_warning": False,
        "reached_max_iter": False,
        "converged_without_warning": np.nan,
        "convergence_message": "",
    }

    saved_coefficients = None

    if use_saved_coefficients_if_possible:
        saved_coefficients = get_saved_coefficients(
            seed_coeffs_df=seed_coeffs_df,
            c0=c0,
            c1=c1,
            feature_cols=feature_cols,
            require_weight_match=require_weight_match,
        )

    if saved_coefficients is None:
        raise FileNotFoundError(
            f"No se encontraron coeficientes guardados para evaluar {model_name} "
            f"en {split_name} (scenario={scenario_name}, seed={seed}, c0={c0}, c1={c1}). "
            "La fase de evaluación no reentrena modelos; ejecuta antes la fase TRAIN."
        )

    intercept = saved_coefficients["intercept"]
    betas = saved_coefficients["betas"]
    convergence_info = saved_coefficients["convergence_info"]

    t_pred_start = time.perf_counter()
    y_prob, y_pred = predict_from_saved_coefficients(
        X_eval_scaled,
        intercept=intercept,
        betas=betas,
    )
    prediction_seconds = time.perf_counter() - t_pred_start
    used_saved_coefficients = True

    result_row = {
        "scenario": scenario_name,
        "seed": seed,
        "model_name": model_name,
        "c0": float(c0),
        "c1": float(c1),
        "n_iter": convergence_info["n_iter"],
        "max_iter": convergence_info["max_iter"],
        "convergence_warning": convergence_info["convergence_warning"],
        "reached_max_iter": convergence_info["reached_max_iter"],
        "converged_without_warning": convergence_info["converged_without_warning"],
        "convergence_message": convergence_info["convergence_message"],
        "used_saved_coefficients": used_saved_coefficients,
    }

    t_metrics_start = time.perf_counter()
    result_row.update(
        compute_metrics_for_split(y_eval, y_pred, y_prob, split_name)
    )
    metrics_seconds = time.perf_counter() - t_metrics_start

    predictions = []
    add_prediction_rows(
        predictions,
        scenario_name=scenario_name,
        seed=seed,
        model_name=model_name,
        split_name=split_name,
        split_df=eval_df,
        y_true=y_eval,
        y_pred=y_pred,
        y_prob=y_prob,
        c0=c0,
        c1=c1,
    )

    total_seconds = time.perf_counter() - t_total_start

    timer_row = {
        "scenario": scenario_name,
        "seed": seed,
        "stage": f"final_{split_name}",
        "model_name": model_name,
        "n_models": 1,
        "scaler_seconds": 0.0,
        "fit_seconds": fit_seconds,
        "validation_prediction_seconds": prediction_seconds if split_name == "val" else 0.0,
        "test_prediction_seconds": prediction_seconds if split_name == "test" else 0.0,
        "metrics_seconds": metrics_seconds,
        "convergence_warnings": int(convergence_info["convergence_warning"]),
        "reached_max_iter_count": int(convergence_info["reached_max_iter"]),
        "n_iter": convergence_info["n_iter"],
        "max_iter": convergence_info["max_iter"],
        "total_seconds": total_seconds,
        "total_time_readable": format_seconds(total_seconds),
        "used_saved_coefficients": used_saved_coefficients,
        "skipped_because_existing_outputs": False,
    }

    return result_row, predictions, timer_row



def get_final_model_config_for_seed(model_name, best_c0, best_c1, y_train):
    """
    Devuelve la configuración de una estrategia final concreta para una seed.
    """
    balanced_c0, balanced_c1 = compute_sklearn_balanced_weights(y_train)

    if model_name == "best_val_f1_grid":
        return {
            "model_name": "best_val_f1_grid",
            "class_weight": {0: float(best_c0), 1: float(best_c1)},
            "c0": float(best_c0),
            "c1": float(best_c1),
            "coefficients_strategy_name": "grid_manual",
            "use_saved_coefficients_if_possible": True,
            "require_weight_match": True,
        }

    if model_name == "equal_weights":
        return {
            "model_name": "equal_weights",
            "class_weight": {0: EQUAL_WEIGHTS_C0, 1: EQUAL_WEIGHTS_C1},
            "c0": EQUAL_WEIGHTS_C0,
            "c1": EQUAL_WEIGHTS_C1,
            "coefficients_strategy_name": "equal_weights",
            "use_saved_coefficients_if_possible": True,
            "require_weight_match": True,
        }

    if model_name == "sklearn_balanced":
        return {
            "model_name": "sklearn_balanced",
            "class_weight": "balanced",
            "c0": balanced_c0,
            "c1": balanced_c1,
            "coefficients_strategy_name": "sklearn_balanced",
            "use_saved_coefficients_if_possible": True,
            "require_weight_match": True,
        }

    raise ValueError(f"Modelo final no reconocido: {model_name}")


def process_one_seed_final_single_model(seed, scenario_name, config, best_c0, best_c1, split_name, model_name):
    """
    Evalúa una única estrategia final en VALIDACIÓN o TEST para una semilla.

    No entrena. Carga coeficientes guardados en TRAIN y calcula predicciones y
    métricas en el split indicado. Esta separación permite medir tiempos reales
    independientes por estrategia.
    """
    t_seed_start = time.perf_counter()

    df = load_or_create_dataset(seed, scenario_name, config)
    feature_cols = [c for c in df.columns if c.startswith("x_")]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    eval_df = df[df["split"] == split_name].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["target"].to_numpy(dtype=int)

    X_eval = eval_df[feature_cols].to_numpy(dtype=float)
    y_eval = eval_df["target"].to_numpy(dtype=int)

    t_scaler_start = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    scaler_seconds = time.perf_counter() - t_scaler_start

    model_config = get_final_model_config_for_seed(
        model_name=model_name,
        best_c0=best_c0,
        best_c1=best_c1,
        y_train=y_train,
    )

    seed_coeffs_df = load_seed_coefficients(
        scenario_name,
        seed,
        model_config["coefficients_strategy_name"],
    )

    result_row, predictions, timer_row = evaluate_one_final_model(
        seed=seed,
        scenario_name=scenario_name,
        split_name=split_name,
        model_name=model_config["model_name"],
        class_weight=model_config["class_weight"],
        c0=model_config["c0"],
        c1=model_config["c1"],
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_eval_scaled=X_eval_scaled,
        y_eval=y_eval,
        eval_df=eval_df,
        feature_cols=feature_cols,
        seed_coeffs_df=seed_coeffs_df,
        use_saved_coefficients_if_possible=model_config["use_saved_coefficients_if_possible"],
        require_weight_match=model_config["require_weight_match"],
    )

    timer_row["scaler_seconds"] = scaler_seconds
    timer_row["total_seconds"] = time.perf_counter() - t_seed_start
    timer_row["total_time_readable"] = format_seconds(timer_row["total_seconds"])

    return pd.DataFrame([result_row]), pd.DataFrame(predictions), pd.DataFrame([timer_row])


# ============================================================
# 5. FUNCIONES PARA FIGURAS CON PINES MEJORADOS
# ============================================================

def value_to_axis_position(value, sorted_values):
    sorted_values = np.asarray(sorted_values, dtype=float)
    value = float(value)

    if value <= 0:
        value = sorted_values[0]

    log_values = np.log10(sorted_values)
    log_value = np.log10(value)

    if log_value <= log_values[0]:
        return 0.0

    if log_value >= log_values[-1]:
        return float(len(sorted_values) - 1)

    return float(np.interp(log_value, log_values, np.arange(len(sorted_values))))


def get_nearest_metric_value(pivot, c0, c1):
    """
    Devuelve el valor REAL de la superficie en el punto de la rejilla más cercano.

    No interpola. Esto es importante porque queremos representar valores reales
    que existen en la matriz de resultados de la rejilla.
    """
    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)

    nearest_c0 = c0_values[
        np.argmin(np.abs(np.log10(c0_values) - np.log10(float(c0))))
    ]

    nearest_c1 = c1_values[
        np.argmin(np.abs(np.log10(c1_values) - np.log10(float(c1))))
    ]

    value = pivot.loc[nearest_c0, nearest_c1]

    if pd.isna(value):
        return 0.0

    return float(value)


def get_exact_grid_metric_value(pivot, c0, c1):
    """
    Devuelve el valor de la métrica en la rejilla para c0 y c1.
    Si c0 y c1 pertenecen exactamente a C_VALUES, devuelve ese valor. Si no pertenecen exactamente, usa el punto real más cercano de la rejilla, sin interpolar.
    
    """
    return get_nearest_metric_value(pivot, c0, c1)


def get_real_model_metric_value(comparison_df, scenario_name, model_name, metric):
    """
    Devuelve el valor real agregado de una métrica para un modelo concreto en la tabla comparativa final de validación.

    Por ejemplo:
    - model_name='best_val_f1_grid'
    - model_name='sklearn_balanced'
    - model_name='equal_weights'

    Si no existe la tabla o la columna, devuelve None.
    """
    if comparison_df is None:
        return None

    if metric not in comparison_df.columns:
        return None

    rows = comparison_df[
        (comparison_df["scenario"] == scenario_name)
        & (comparison_df["model_name"] == model_name)
    ]

    if rows.empty:
        return None

    value = rows.iloc[0][metric]

    if pd.isna(value):
        return None

    return float(value)


def get_pin_points(best_df, balanced_df, scenario_name):
    """
    Devuelve los tres puntos de referencia que se dibujan en las figuras:

    1) Best F1:
       pesos seleccionados por maximizar val_f1_mean.

    2) Balanced:
       pesos equivalentes a class_weight="balanced" de Scikit-Learn.

    3) Equal weights:
       punto sin ponderación diferencial entre clases, equivalente a c0=1 y c1=1.
    """
    best_row = best_df[best_df["scenario"] == scenario_name].iloc[0]
    balanced_row = balanced_df[balanced_df["scenario"] == scenario_name].iloc[0]

    best_point = {
        "label": "Best val F1",
        "short_label": "Best F1",
        "model_name": "best_val_f1_grid",
        "c0": float(best_row["c0"]),
        "c1": float(best_row["c1"]),
        "is_grid_point": True,
    }

    balanced_point = {
        "label": "Sklearn balanced",
        "short_label": "Sklearn balanced",
        "model_name": "sklearn_balanced",
        "c0": float(balanced_row["balanced_c0_mean"]),
        "c1": float(balanced_row["balanced_c1_mean"]),
        "is_grid_point": False,
    }

    equal_is_grid_point = (
        any(np.isclose(np.asarray(C_VALUES, dtype=float), EQUAL_WEIGHTS_C0))
        and any(np.isclose(np.asarray(C_VALUES, dtype=float), EQUAL_WEIGHTS_C1))
    )

    equal_point = {
        "label": "Equal weights",
        "short_label": "Equal weights",
        "model_name": "equal_weights",
        "c0": EQUAL_WEIGHTS_C0,
        "c1": EQUAL_WEIGHTS_C1,
        "is_grid_point": equal_is_grid_point,
    }

    return best_point, balanced_point, equal_point


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def format_weights_for_pin_text(point):
    """
    Formatea los pesos escritos junto al pin.
    Se muestran como c0 y c1 porque, con flip_y=0.0 y split estratificado, los pesos de Sklearn balanced son constantes entre seeds.
    """
    return f"c0={point['c0']:.3g}, c1={point['c1']:.3g}"


def format_pin_text(point, metric_value):
    """
    Texto de las etiquetas 2D.
    """
    label = point.get("short_label", point.get("label", "Punto"))

    return (
        f"{label}: {metric_value:.3f}\n"
        f"{format_weights_for_pin_text(point)}"
    )


def estimate_label_box_size():
    """
    Tamaño aproximado de las cajas de texto en coordenadas de la rejilla.

    El tamaño se adapta al número de valores de C_VALUES. Así, si se añaden o
    se quitan puntos de la rejilla, el algoritmo sigue dejando margen suficiente
    para que los recuadros no se corten con los bordes ni tapen las estrellas.
    """
    grid_span = max(1.0, float(len(C_VALUES) - 1))

    # La anchura real del texto en coordenadas de datos crece con el tamaño de
    # la rejilla representada. Se usa una estimación conservadora para evitar
    # que el borde derecho del recuadro se corte en puntos extremos.
    box_width = clamp(0.29 * grid_span, 4.80, 7.20)
    box_height = clamp(0.075 * grid_span, 1.35, 1.90)

    return box_width, box_height


def make_label_rect(text_x, text_y, box_width, box_height):
    """
    Rectángulo aproximado de una etiqueta.

    Las etiquetas 2D se dibujan con ha='left' y va='center'. Por tanto, text_x
    representa el borde izquierdo aproximado del recuadro y text_y su centro
    vertical.
    """
    return (
        text_x,
        text_x + box_width,
        text_y - box_height / 2,
        text_y + box_height / 2,
    )


def rectangles_overlap(rect_a, rect_b, padding=0.34):
    """
    Comprueba si dos rectángulos se solapan, usando un margen extra.
    """
    ax0, ax1, ay0, ay1 = rect_a
    bx0, bx1, by0, by1 = rect_b

    return not (
        ax1 + padding < bx0
        or bx1 + padding < ax0
        or ay1 + padding < by0
        or by1 + padding < ay0
    )


def overlap_area(rect_a, rect_b):
    """
    Área de solape entre dos rectángulos aproximados.
    """
    ax0, ax1, ay0, ay1 = rect_a
    bx0, bx1, by0, by1 = rect_b

    overlap_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_y = max(0.0, min(ay1, by1) - max(ay0, by0))

    return overlap_x * overlap_y


def point_inside_rect(point_x, point_y, rect, padding=0.58):
    """
    Evita que una caja de texto tape un marcador.
    """
    x0, x1, y0, y1 = rect
    return (
        x0 - padding <= point_x <= x1 + padding
        and y0 - padding <= point_y <= y1 + padding
    )


def preferred_direction_penalty(preferred, x, y, rect):
    """
    Penalización suave para mantener la etiqueta en una zona visual coherente.
    """
    x0, x1, y0, y1 = rect
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2

    if preferred == "up" and center_y < y:
        return 1.20
    if preferred == "down" and center_y > y:
        return 1.20
    if preferred == "right" and center_x < x:
        return 1.20
    if preferred == "left" and center_x > x:
        return 1.20

    return 0.0


def edge_margin_penalty(rect, max_x, max_y):
    """
    Penaliza recuadros demasiado pegados al borde de la figura.

    Esto evita casos como etiquetas en la esquina superior derecha que quedan
    visualmente cortadas o demasiado pegadas al marco del eje.
    """
    x0, x1, y0, y1 = rect
    margin_x = max(0.35, 0.025 * max_x)
    margin_y = max(0.35, 0.025 * max_y)

    penalty = 0.0
    penalty += max(0.0, margin_x - x0) * 9.0
    penalty += max(0.0, x1 - (max_x - margin_x)) * 9.0
    penalty += max(0.0, margin_y - y0) * 9.0
    penalty += max(0.0, y1 - (max_y - margin_y)) * 9.0
    return penalty


def generate_label_candidates(x, y, max_x, max_y, preferred="up"):
    """
    Genera posiciones candidatas para una etiqueta 2D.

    El cálculo es relativo al tamaño real de la rejilla, por lo que funciona con
    17, 25 o cualquier otro número de valores en C_VALUES. Se priorizan cajas
    cercanas, pero dejando una flecha visible y evitando bordes, estrellas y
    solapes.
    """
    box_width, box_height = estimate_label_box_size()

    min_tx = 0.35
    max_tx = max(0.35, max_x - box_width - 0.35)
    min_ty = box_height / 2 + 0.35
    max_ty = max(box_height / 2 + 0.35, max_y - box_height / 2 - 0.35)

    gap_x = clamp(0.070 * max_x, 1.25, 1.80)
    gap_y = clamp(0.055 * max_y, 1.05, 1.45)

    # Si el punto está cerca de un borde, se fuerza de forma natural que el
    # recuadro mire hacia el interior del panel.
    edge_preferred = preferred
    if x > max_x - (box_width + gap_x + 0.60):
        edge_preferred = "left"
    elif x < box_width + gap_x + 0.60:
        edge_preferred = "right"

    if y > max_y - (box_height + gap_y + 0.60) and edge_preferred not in ["left", "right"]:
        edge_preferred = "down"
    elif y < box_height + gap_y + 0.60 and edge_preferred not in ["left", "right"]:
        edge_preferred = "up"

    local_offsets = []

    # Primer anillo: posiciones claras alrededor del punto.
    local_offsets.extend([
        (gap_x, 0.00),
        (-box_width - gap_x, 0.00),
        (gap_x, gap_y),
        (gap_x, -gap_y),
        (-box_width - gap_x, gap_y),
        (-box_width - gap_x, -gap_y),
        (-box_width / 2, box_height / 2 + gap_y),
        (-box_width / 2, -box_height / 2 - gap_y),
    ])

    # Segundo anillo: por si hay dos o tres etiquetas próximas.
    local_offsets.extend([
        (gap_x + 0.75, gap_y + 0.65),
        (gap_x + 0.75, -gap_y - 0.65),
        (-box_width - gap_x - 0.75, gap_y + 0.65),
        (-box_width - gap_x - 0.75, -gap_y - 0.65),
        (-box_width / 2, box_height / 2 + gap_y + 1.00),
        (-box_width / 2, -box_height / 2 - gap_y - 1.00),
    ])

    if edge_preferred == "up":
        local_offsets = sorted(local_offsets, key=lambda t: (t[1] < 0, abs(t[0]) + abs(t[1])))
    elif edge_preferred == "down":
        local_offsets = sorted(local_offsets, key=lambda t: (t[1] > 0, abs(t[0]) + abs(t[1])))
    elif edge_preferred == "right":
        local_offsets = sorted(local_offsets, key=lambda t: (t[0] < 0, abs(t[0]) + abs(t[1])))
    elif edge_preferred == "left":
        local_offsets = sorted(local_offsets, key=lambda t: (t[0] > 0, abs(t[0]) + abs(t[1])))

    candidates = []
    seen = set()

    def add_candidate(tx, ty, fallback=False):
        tx = clamp(tx, min_tx, max_tx)
        ty = clamp(ty, min_ty, max_ty)

        key = (round(tx, 3), round(ty, 3))
        if key in seen:
            return
        seen.add(key)

        rect = make_label_rect(tx, ty, box_width, box_height)
        center_x = tx + box_width / 2
        center_y = ty
        anchor_x, anchor_y = nearest_point_on_rect(x, y, rect)

        distance_to_center = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
        distance_to_edge = np.sqrt((anchor_x - x) ** 2 + (anchor_y - y) ** 2)

        # El término principal es la distancia al borde del recuadro, porque es
        # la longitud visual de la flecha. Se penaliza que sea demasiado corta.
        score = 2.10 * distance_to_edge + 0.20 * distance_to_center
        score += preferred_direction_penalty(edge_preferred, x, y, rect)
        score += edge_margin_penalty(rect, max_x, max_y)

        if distance_to_edge < 0.85:
            score += 18.0

        if fallback:
            score += 22.0

        candidates.append({
            "text_x": tx,
            "text_y": ty,
            "rect": rect,
            "score": score,
        })

    for dx, dy in local_offsets:
        add_candidate(x + dx, y + dy, fallback=False)

    # Candidatos de respaldo repartidos por el panel. Solo ganan si los puntos
    # están en una configuración muy complicada.
    fallback_x_values = np.linspace(min_tx, max_tx, 6)
    fallback_y_values = np.linspace(min_ty, max_ty, 6)

    for tx in fallback_x_values:
        for ty in fallback_y_values:
            add_candidate(tx, ty, fallback=True)

    candidates = sorted(candidates, key=lambda c: c["score"])
    return candidates


def choose_text_positions_for_all_pins(points, max_x, max_y):
    """
    Elige conjuntamente las posiciones de todos los recuadros 2D.

    Prueba combinaciones de candidatos y evita: recuadros cortados, solapes
    entre recuadros, recuadros encima de estrellas y flechas excesivamente
    cortas o largas. No depende de una rejilla concreta.
    """
    all_candidates = [
        generate_label_candidates(
            item["x"],
            item["y"],
            max_x,
            max_y,
            preferred=item.get("preferred", "up"),
        )
        for item in points
    ]

    point_positions = [(item["x"], item["y"]) for item in points]

    best_combo = None
    best_score = np.inf
    best_soft_combo = None
    best_soft_score = np.inf

    for combo in product(*all_candidates):
        hard_overlap = False
        soft_penalty = 0.0

        for i in range(len(combo)):
            rect_i = combo[i]["rect"]

            for point_x, point_y in point_positions:
                if point_inside_rect(point_x, point_y, rect_i):
                    hard_overlap = True
                    soft_penalty += 3000.0

            for j in range(i + 1, len(combo)):
                rect_j = combo[j]["rect"]

                if rectangles_overlap(rect_i, rect_j):
                    hard_overlap = True
                    soft_penalty += 3000.0 + 300.0 * overlap_area(rect_i, rect_j)

        distance_penalty = 0.0
        for item, candidate in zip(points, combo):
            rect = candidate["rect"]
            anchor_x, anchor_y = nearest_point_on_rect(item["x"], item["y"], rect)
            connector_distance = np.sqrt((anchor_x - item["x"]) ** 2 + (anchor_y - item["y"]) ** 2)
            distance_penalty += 1.10 * connector_distance
            if connector_distance < 0.85:
                soft_penalty += 800.0

        score = sum(candidate["score"] for candidate in combo) + distance_penalty
        soft_score = score + soft_penalty

        if soft_score < best_soft_score:
            best_soft_score = soft_score
            best_soft_combo = combo

        if not hard_overlap and score < best_score:
            best_score = score
            best_combo = combo

    selected_combo = best_combo if best_combo is not None else best_soft_combo

    for item, candidate in zip(points, selected_combo):
        item["text_x"] = candidate["text_x"]
        item["text_y"] = candidate["text_y"]
        item["text_rect"] = candidate["rect"]

    return points


def nearest_point_on_rect(point_x, point_y, rect):
    """
    Devuelve el punto del borde del recuadro más cercano al marcador.
    Así la línea de referencia no va siempre al borde izquierdo de la caja, sino al borde realmente más cercano. Esto acorta la línea y mejora la lectura visual.
    
    """
    x0, x1, y0, y1 = rect
    nearest_x = clamp(point_x, x0, x1)
    nearest_y = clamp(point_y, y0, y1)

    # Si el punto quedara dentro del rectángulo por cualquier motivo, usamos
    # el borde más cercano para que la línea siga siendo coherente.
    if x0 < point_x < x1 and y0 < point_y < y1:
        distances = {
            "left": abs(point_x - x0),
            "right": abs(x1 - point_x),
            "bottom": abs(point_y - y0),
            "top": abs(y1 - point_y),
        }
        side = min(distances, key=distances.get)
        if side == "left":
            nearest_x = x0
            nearest_y = point_y
        elif side == "right":
            nearest_x = x1
            nearest_y = point_y
        elif side == "bottom":
            nearest_x = point_x
            nearest_y = y0
        else:
            nearest_x = point_x
            nearest_y = y1

    return nearest_x, nearest_y


def draw_contrast_reference_line(ax, x0, y0, x1, y1, color):
    """
    Dibuja una línea simple desde el borde del recuadro hasta la estrella.

    No usa punta de flecha ni triángulos. Se dibujan dos líneas superpuestas:
    una blanca más gruesa como halo y otra encima con el color del modelo.
    Así la conexión se ve sobre fondos claros y oscuros sin tapar demasiado
    la estrella ni el recuadro.
    """
    length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    if length < 0.12:
        return

    # Se recortan ligeramente los extremos para que la línea no atraviese el
    # centro de la estrella ni se meta dentro del recuadro. Es solo visual.
    ux = (x1 - x0) / length
    uy = (y1 - y0) / length

    marker_gap = min(0.34, 0.22 * length)
    box_gap = min(0.12, 0.10 * length)

    start_x = x0 + ux * marker_gap
    start_y = y0 + uy * marker_gap
    end_x = x1 - ux * box_gap
    end_y = y1 - uy * box_gap

    ax.plot(
        [start_x, end_x],
        [start_y, end_y],
        color="white",
        linewidth=4.0,
        alpha=0.98,
        solid_capstyle="round",
        zorder=24,
        clip_on=False,
    )

    ax.plot(
        [start_x, end_x],
        [start_y, end_y],
        color=color,
        linewidth=1.65,
        alpha=0.98,
        solid_capstyle="round",
        zorder=25,
        clip_on=False,
    )


def nearest_point_on_text_bbox(ax, text_artist, point_x, point_y):
    """
    Calcula el punto real del borde del recuadro de texto más cercano al
    marcador usando la caja renderizada por Matplotlib.

    Esto es más preciso que usar una caja estimada, porque tiene en cuenta el
    tamaño real del texto y del bbox. Así la línea llega realmente al recuadro.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_display = text_artist.get_window_extent(renderer=renderer).expanded(1.04, 1.12)

    inv = ax.transData.inverted()
    (x0, y0) = inv.transform((bbox_display.x0, bbox_display.y0))
    (x1, y1) = inv.transform((bbox_display.x1, bbox_display.y1))

    # Asegura orden correcto por si alguna transformación invierte ejes.
    rect = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    return nearest_point_on_rect(point_x, point_y, rect)



def move_text_artist_inside_axes(ax, text_artist, pad_pixels=6):
    """
    Ajusta un recuadro ya dibujado para que no quede cortado por los bordes.

    La primera colocación se hace en coordenadas de rejilla; después se mide el
    tamaño real renderizado por Matplotlib y se corrige en píxeles. Esto evita
    que un cambio en C_VALUES o en el texto haga que el recuadro se salga del
    panel.
    """
    fig = ax.figure

    for _ in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_bbox = text_artist.get_window_extent(renderer=renderer)
        axes_bbox = ax.get_window_extent(renderer=renderer)

        dx_pixels = 0.0
        dy_pixels = 0.0

        if text_bbox.x0 < axes_bbox.x0 + pad_pixels:
            dx_pixels += (axes_bbox.x0 + pad_pixels) - text_bbox.x0
        if text_bbox.x1 > axes_bbox.x1 - pad_pixels:
            dx_pixels -= text_bbox.x1 - (axes_bbox.x1 - pad_pixels)
        if text_bbox.y0 < axes_bbox.y0 + pad_pixels:
            dy_pixels += (axes_bbox.y0 + pad_pixels) - text_bbox.y0
        if text_bbox.y1 > axes_bbox.y1 - pad_pixels:
            dy_pixels -= text_bbox.y1 - (axes_bbox.y1 - pad_pixels)

        if abs(dx_pixels) < 0.5 and abs(dy_pixels) < 0.5:
            break

        current_x, current_y = text_artist.get_position()
        current_display = ax.transData.transform((current_x, current_y))
        new_display = (current_display[0] + dx_pixels, current_display[1] + dy_pixels)
        new_x, new_y = ax.transData.inverted().transform(new_display)
        text_artist.set_position((new_x, new_y))

    return text_artist


def get_label_metric_value_for_pin(point, pivot, metric, scenario_name, val_comparison_df=None):
    """
    Valor que se escribe en la etiqueta/leyenda.

    - Para Best F1, el valor sale de la rejilla, porque es un punto de la rejilla.
    - Para Sklearn balanced, el valor real sale de la tabla comparativa final
      de validación si está disponible. Si no, se usa el punto más cercano
      de la rejilla sin interpolar como respaldo.
    - Para Equal weights, el valor se toma de la tabla comparativa final
      si está disponible; si no, se toma del punto (1,1) de la rejilla.
    """
    if point.get("model_name") in ["sklearn_balanced", "equal_weights"]:
        real_value = get_real_model_metric_value(
            val_comparison_df,
            scenario_name,
            point.get("model_name"),
            metric,
        )
        if real_value is not None:
            return real_value

    return get_exact_grid_metric_value(pivot, point["c0"], point["c1"])



def draw_single_star_marker_2d(ax, x, y, color, size, angle_degrees=0.0, zorder=32):
    """
    Dibuja una única estrella 2D en el punto real (x, y).

    La rotación solo afecta al símbolo visual; no modifica la coordenada.
    """
    marker = (5, 1, angle_degrees)
    marker_size = max(7.0, np.sqrt(float(size)) * 1.12)

    ax.plot(
        [x],
        [y],
        linestyle="None",
        marker=marker,
        markersize=marker_size,
        markerfacecolor=color,
        markeredgecolor=PIN_EDGE_COLOR,
        markeredgewidth=1.00,
        zorder=zorder,
    )


def draw_2d_stars_with_overlap_control(ax, points):
    """
    Dibuja las estrellas 2D controlando solapes.

    Si varias referencias caen exactamente en el mismo punto del mapa 2D, se
    mantienen en esa misma coordenada, pero la estrella de detrás se rota y se
    hace un poco mayor para que se distingan sus puntas.
    """
    groups = {}

    for item in points:
        key = (
            round(float(item["x"]), 2),
            round(float(item["y"]), 2),
        )
        groups.setdefault(key, []).append(item)

    for group_items in groups.values():
        n_items = len(group_items)

        if n_items == 1:
            item = group_items[0]
            draw_single_star_marker_2d(
                ax,
                item["x"],
                item["y"],
                item["color"],
                item["size"],
                angle_degrees=0.0,
                zorder=32,
            )
            continue

        if n_items == 2:
            angles = [-35.0, 0.0]
            factors = [1.00, 1.00]
        elif n_items == 3:
            angles = [-40.0, 0.0, 40.0]
            factors = [1.00, 1.00, 1.00]
        else:
            angles = np.linspace(-42.0, 42.0, n_items).tolist()
            factors = [1.00 for _ in range(n_items)]

        for idx, (item, angle, factor) in enumerate(zip(group_items, angles, factors)):
            draw_single_star_marker_2d(
                ax,
                item["x"],
                item["y"],
                item["color"],
                item["size"] * factor,
                angle_degrees=angle,
                zorder=32 + idx,
            )


def add_pins_to_heatmap(ax, pivot, best_point, balanced_point, metric, scenario_name, val_comparison_df=None, equal_point=None):
    """
    Añade pines y etiquetas compactas en 2D.

    Las posiciones se recalculan automáticamente a partir del tamaño real de la
    rejilla, por lo que funcionan aunque se añadan o se quiten valores en
    C_VALUES. Las cajas se colocan de forma conjunta para evitar solapes entre
    etiquetas y para no tapar las estrellas.

    Las flechas se dibujan después de renderizar las cajas, usando el borde real
    del recuadro. Así quedan cortas, llegan al punto correcto y se ven bien sobre
    fondos claros u oscuros gracias al halo blanco.
    """
    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)

    max_x = len(c1_values) - 1
    max_y = len(c0_values) - 1

    best_x = value_to_axis_position(best_point["c1"], c1_values)
    best_y = value_to_axis_position(best_point["c0"], c0_values)

    balanced_x = value_to_axis_position(balanced_point["c1"], c1_values)
    balanced_y = value_to_axis_position(balanced_point["c0"], c0_values)

    points = [
        {
            "point": best_point,
            "color": MODEL_COLORS["best_val_f1_grid"],
            "x": best_x,
            "y": best_y,
            "preferred": "up",
            "size": 125,
        },
        {
            "point": balanced_point,
            "color": MODEL_COLORS["sklearn_balanced"],
            "x": balanced_x,
            "y": balanced_y,
            "preferred": "down",
            "size": 125,
        },
    ]

    if equal_point is not None:
        equal_x = value_to_axis_position(equal_point["c1"], c1_values)
        equal_y = value_to_axis_position(equal_point["c0"], c0_values)

        points.append({
            "point": equal_point,
            "color": MODEL_COLORS["equal_weights"],
            "x": equal_x,
            "y": equal_y,
            "preferred": "right",
            "size": 125,
        })

    points = choose_text_positions_for_all_pins(points, max_x, max_y)

    # Primero se dibujan las estrellas. Si coinciden, se mantienen en la misma
    # coordenada y se diferencian por rotación visual.
    draw_2d_stars_with_overlap_control(ax, points)

    text_items = []

    # Después se dibujan los recuadros. Las flechas se añaden en un segundo
    # paso, usando la caja real ya renderizada.
    for item in points:
        point = item["point"]

        metric_value = get_label_metric_value_for_pin(
            point,
            pivot,
            metric,
            scenario_name,
            val_comparison_df,
        )

        text_artist = ax.text(
            item["text_x"],
            item["text_y"],
            format_pin_text(point, metric_value),
            fontsize=7.0,
            color=PIN_TEXT_COLOR,
            ha="left",
            va="center",
            zorder=30,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.16",
                fc="white",
                ec=PIN_EDGE_COLOR,
                alpha=0.94,
                linewidth=0.85,
            ),
        )

        text_items.append((item, text_artist))

    # Ajuste final con la caja real renderizada. Evita que un recuadro quede
    # cortado por el borde cuando se modifica la rejilla o cambia el texto.
    for _, text_artist in text_items:
        move_text_artist_inside_axes(ax, text_artist, pad_pixels=6)

    # Flechas desde el borde real del recuadro hasta la estrella. El cálculo del
    # borde se hace con el bbox renderizado, no con una estimación.
    for item, text_artist in text_items:
        text_anchor_x, text_anchor_y = nearest_point_on_text_bbox(
            ax,
            text_artist,
            item["x"],
            item["y"],
        )

        draw_contrast_reference_line(
            ax,
            item["x"],
            item["y"],
            text_anchor_x,
            text_anchor_y,
            item["color"],
        )


def add_subtle_grid_points_2d(ax, pivot):
    """
    Dibuja una rejilla muy sutil de puntos sobre los mapas 2D.

    Cada punto pequeño representa una combinación real evaluada de la rejilla
    manual (c0, c1). El fondo del mapa se dibuja con interpolación visual para
    suavizar la figura, por lo que estos puntos ayudan a distinguir qué valores
    proceden directamente de la evaluación y cuáles son transición visual.
    """
    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)

    x_pos = np.arange(len(c1_values))
    y_pos = np.arange(len(c0_values))
    X_grid, Y_grid = np.meshgrid(x_pos, y_pos)

    ax.scatter(
        X_grid.ravel(),
        Y_grid.ravel(),
        s=12,
        c="white",
        alpha=0.30,
        marker=".",
        edgecolors="none",
        zorder=8,
    )


def add_subtle_grid_points_3d(ax, X_grid, Y_grid, Z):
    """
    Dibuja puntos discretos y casi transparentes sobre la superficie 3D.

    Cada punto corresponde a un valor real de la rejilla manual evaluada. La
    superficie 3D une visualmente esos valores, por lo que los puntos aclaran
    cuáles son las posiciones efectivamente calculadas.
    """
    ax.scatter(
        X_grid.ravel(),
        Y_grid.ravel(),
        Z.ravel(),
        s=14,
        c="white",
        alpha=0.30,
        marker=".",
        edgecolors="none",
        depthshade=False,
        zorder=230,
    )


def add_2d_grid_note(fig):
    """
    Añade una nota breve en los paneles 2D para explicar la rejilla de puntos.
    """
    fig.text(
        0.50,
        0.012,
        "Puntos blancos sutiles: combinaciones reales evaluadas de la rejilla manual; el fondo está suavizado/interpolado visualmente entre ellas.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#444444",
    )


def draw_single_star_marker_3d(ax, x, y, z, color, angle_degrees=0.0, size=12.5, zorder=1000):
    """
    Dibuja una única estrella 3D en el punto real (x, y, z).

    Se usa ax.plot con marcador rotado, porque en la práctica la rotación del
    símbolo se aprecia mejor así en 3D. La coordenada no se modifica.
    """
    marker = (5, 1, angle_degrees)

    ax.plot(
        [x],
        [y],
        [z],
        linestyle="None",
        marker=marker,
        markersize=size,
        markerfacecolor=color,
        markeredgecolor=PIN_EDGE_COLOR,
        markeredgewidth=1.00,
        zorder=zorder,
    )


def draw_3d_stars_with_overlap_control(ax, star_items):
    """
    Dibuja las estrellas 3D controlando posibles solapes.

    Si varias referencias caen en el mismo punto visual de la rejilla (mismo
    c0 y c1), se mantienen en esa misma coordenada, pero la estrella posterior
    se rota y se hace un poco mayor para que se aprecien sus puntas.
    """
    groups = {}

    for item in star_items:
        key = (
            round(float(item["x"]), 2),
            round(float(item["y"]), 2),
        )
        groups.setdefault(key, []).append(item)

    for group_items in groups.values():
        n_items = len(group_items)

        if n_items == 1:
            item = group_items[0]
            draw_single_star_marker_3d(
                ax,
                item["x"],
                item["y"],
                item["z"],
                item["color"],
                angle_degrees=0.0,
                size=12.5,
                zorder=1000,
            )
            continue

        if n_items == 2:
            angles = [-35.0, 0.0]
            sizes = [12.5, 12.5]
        elif n_items == 3:
            angles = [-40.0, 0.0, 40.0]
            sizes = [12.5, 12.5, 12.5]
        else:
            angles = np.linspace(-42.0, 42.0, n_items).tolist()
            sizes = [12.5 for _ in range(n_items)]

        for idx, (item, angle, size) in enumerate(zip(group_items, angles, sizes)):
            draw_single_star_marker_3d(
                ax,
                item["x"],
                item["y"],
                item["z"],
                item["color"],
                angle_degrees=angle,
                size=size,
                zorder=1000 + idx,
            )


def add_pins_to_3d_surface(ax, pivot, best_point, balanced_point, metric, scenario_name, val_comparison_df=None, equal_point=None):
    """
    Añade pines en 3D usando valores reales, sin interpolación.

    Criterio aplicado:
    - Best F1 pertenece a la rejilla manual, por tanto se dibuja en el punto
      real correspondiente a su c0 y c1.
    - Sklearn balanced normalmente no pertenece exactamente a la rejilla manual.
      Se dibuja en su posición real c0-c1 y con su valor real tomado de la
      tabla comparativa de validación si está disponible. Por eso puede no
      coincidir visualmente con un nodo concreto de la superficie 3D.
    - Equal weights se trata como estrategia final independiente. Si (1,1)
      pertenece a C_VALUES coincide con un nodo de la rejilla; si se elimina
      de la rejilla, se sigue representando con su métrica real final.

    Si dos referencias se superponen, no se mueven. La estrella posterior se
    rota y se dibuja algo mayor para que se identifiquen ambos símbolos.
    """
    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)

    pin_info = []
    star_items = []

    best_x = value_to_axis_position(best_point["c1"], c1_values)
    best_y = value_to_axis_position(best_point["c0"], c0_values)
    best_z = get_exact_grid_metric_value(
        pivot,
        best_point["c0"],
        best_point["c1"],
    )

    star_items.append({
        "x": best_x,
        "y": best_y,
        "z": best_z,
        "color": MODEL_COLORS["best_val_f1_grid"],
    })

    pin_info.append({
        "label": best_point.get("short_label", best_point.get("label", "Best F1")),
        "model_name": best_point.get("model_name"),
        "c0": best_point["c0"],
        "c1": best_point["c1"],
        "metric_value": best_z,
        "surface_reference": best_z,
        "color": MODEL_COLORS["best_val_f1_grid"],
        "marker": "*",
        "grid_relation": "punto de la rejilla manual",
    })

    balanced_x = value_to_axis_position(balanced_point["c1"], c1_values)
    balanced_y = value_to_axis_position(balanced_point["c0"], c0_values)
    balanced_surface_ref = get_exact_grid_metric_value(
        pivot,
        balanced_point["c0"],
        balanced_point["c1"],
    )

    balanced_z = get_label_metric_value_for_pin(
        balanced_point,
        pivot,
        metric,
        scenario_name,
        val_comparison_df,
    )

    star_items.append({
        "x": balanced_x,
        "y": balanced_y,
        "z": balanced_z,
        "color": MODEL_COLORS["sklearn_balanced"],
    })

    pin_info.append({
        "label": balanced_point.get("short_label", balanced_point.get("label", "Sklearn balanced")),
        "model_name": balanced_point.get("model_name"),
        "c0": balanced_point["c0"],
        "c1": balanced_point["c1"],
        "metric_value": balanced_z,
        "surface_reference": balanced_surface_ref,
        "color": MODEL_COLORS["sklearn_balanced"],
        "marker": "*",
        "grid_relation": "referencia externa; no suele coincidir con un nodo de la rejilla",
    })

    if equal_point is not None:
        equal_x = value_to_axis_position(equal_point["c1"], c1_values)
        equal_y = value_to_axis_position(equal_point["c0"], c0_values)
        equal_z_surface = get_exact_grid_metric_value(
            pivot,
            equal_point["c0"],
            equal_point["c1"],
        )

        equal_z = get_label_metric_value_for_pin(
            equal_point,
            pivot,
            metric,
            scenario_name,
            val_comparison_df,
        )

        star_items.append({
            "x": equal_x,
            "y": equal_y,
            "z": equal_z,
            "color": MODEL_COLORS["equal_weights"],
        })

        pin_info.append({
            "label": equal_point.get("short_label", equal_point.get("label", "Equal weights")),
            "model_name": equal_point.get("model_name"),
            "c0": equal_point["c0"],
            "c1": equal_point["c1"],
            "metric_value": equal_z,
            "surface_reference": equal_z_surface,
            "color": MODEL_COLORS["equal_weights"],
            "marker": "*",
            "grid_relation": "estrategia final independiente; coincide con la rejilla solo si (1,1) está en C_VALUES",
        })

    draw_3d_stars_with_overlap_control(ax, star_items)

    return pin_info


def add_3d_pin_legend(fig, metric, pin_info):
    """
    Leyenda externa para figuras 3D.

    Incluye símbolo, color, valor real de métrica, c0 y c1. También aclara que
    los puntos pequeños son los nodos reales de la rejilla manual y que la
    superficie une visualmente esos puntos evaluados.
    """
    handles = []

    # Punto sutil de rejilla: representa los nodos reales evaluados.
    handles.append(
        Line2D(
            [0],
            [0],
            marker=".",
            linestyle="None",
            markersize=12.5,
            markerfacecolor="#bfbfbf",
            markeredgecolor="#bfbfbf",
            markeredgewidth=0.0,
            alpha=0.75,
            label="Puntos: rejilla manual evaluada (c0, c1)",
        )
    )

    handles.append(
        Line2D(
            [0],
            [0],
            color="#666666",
            linewidth=2.0,
            alpha=0.65,
            label="Superficie: interpolación visual",
        )
    )

    for info in pin_info:
        weights_text = f"c0={info['c0']:.3g}, c1={info['c1']:.3g}"

        label = (
            f"{info['label']}: {info['metric_value']:.3f} "
            f"| {weights_text}"
        )

        if info.get("model_name") == "sklearn_balanced":
            label += "\nref. externa a rejilla; class_weight=\'balanced\'"

        handle = Line2D(
            [0],
            [0],
            marker=info["marker"],
            linestyle="None",
            markersize=11.8,
            markerfacecolor=info["color"],
            markeredgecolor=PIN_EDGE_COLOR,
            markeredgewidth=1.05,
            label=label,
        )

        handles.append(handle)

    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.055),
        fontsize=9.6,
        frameon=True,
        framealpha=0.96,
        ncol=1,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=1.5,
        handletextpad=0.6,
        title=f"Leyenda 3D ({format_metric_label_for_plot(metric, include_split=True)})",
        title_fontsize=10.4,
    )

def save_3d_surface_plot(agg_df, scenario_name, metric, filename, best_point=None, balanced_point=None, val_comparison_df=None, equal_point=None):
    temp_df = agg_df[agg_df["scenario"] == scenario_name].copy()
    pivot = temp_df.pivot(index="c0", columns="c1", values=metric)

    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)

    x_pos = np.arange(len(c1_values))
    y_pos = np.arange(len(c0_values))

    X_grid, Y_grid = np.meshgrid(x_pos, y_pos)
    Z = pivot.values.astype(float)

    fig = plt.figure(figsize=(12.4, 8.9))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    surf = ax.plot_surface(
        X_grid,
        Y_grid,
        Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.65,
        antialiased=True,
        zorder=1,
    )

    # Puntos reales evaluados de la rejilla manual. La superficie une visualmente
    # estos valores, por eso se añaden puntos pequeños y casi transparentes.
    add_subtle_grid_points_3d(ax, X_grid, Y_grid, Z)

    if best_point is not None and balanced_point is not None:
        pin_info = add_pins_to_3d_surface(
            ax,
            pivot,
            best_point,
            balanced_point,
            metric,
            scenario_name,
            val_comparison_df,
            equal_point=equal_point,
        )
        add_3d_pin_legend(fig, metric, pin_info)

    ax.set_xlabel("c1", labelpad=32)
    ax.set_ylabel("c0", labelpad=16)
    ax.set_zlabel("")

    ax.set_title("")
    fig.suptitle(
        f"{format_scenario_label_for_plot(scenario_name)} · {format_metric_label_for_plot(metric, include_split=True)} · Superficie 3D",
        y=0.965,
        fontsize=13,
    )

    ax.view_init(elev=31, azim=-56)

    step = 2

    ax.set_xticks(x_pos[::step])
    ax.set_yticks(y_pos[::step])

    ax.set_xticklabels(
        [str(x) for x in c1_values[::step]],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_yticklabels(
        [str(y) for y in c0_values[::step]],
        fontsize=8,
    )

    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    ax.tick_params(axis="z", pad=6)

    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    z_range = z_max - z_min if z_max > z_min else 1.0

    # Incluye margen superior para que los marcadores y la superficie se vean correctamente.
    max_pin_z = z_max
    if best_point is not None and balanced_point is not None:
        points_for_zlim = [best_point, balanced_point]

        if equal_point is not None:
            points_for_zlim.append(equal_point)

        for point in points_for_zlim:
            pin_value = get_label_metric_value_for_pin(
                point,
                pivot,
                metric,
                scenario_name,
                val_comparison_df,
            )
            max_pin_z = max(max_pin_z, pin_value)

    ax.set_zlim(0, max_pin_z + 0.04 * z_range)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.72, aspect=18, pad=0.065)
    cbar.set_label(format_metric_label_for_plot(metric, include_split=True), rotation=90, labelpad=12)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.tick_params(labelright=True, labelleft=False, right=True, left=False)

    plt.subplots_adjust(left=0.03, right=0.85, bottom=0.12, top=0.92)

    # Desplaza la barra de color hacia arriba después de ajustar los márgenes.
    # Si se hace antes de subplots_adjust, Matplotlib puede volver a colocarla.
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([
        cbar_pos.x0,
        cbar_pos.y0 + 0.07,
        cbar_pos.width,
        cbar_pos.height,
    ])

    fig.savefig(get_scenario_figure_path(scenario_name, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_panel(agg_df, scenario_name, metrics_panel, title, filename, best_point=None, balanced_point=None, val_comparison_df=None, equal_point=None):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    axes = axes.ravel()

    temp_df = agg_df[agg_df["scenario"] == scenario_name].copy()

    for ax, metric in zip(axes, metrics_panel):
        pivot = temp_df.pivot(index="c0", columns="c1", values=metric)

        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            interpolation="hanning",
        )

        # Puntos reales evaluados de la rejilla manual. El mapa suaviza/interpola
        # visualmente entre ellos, por eso estos puntos aclaran dónde hay datos reales.
        add_subtle_grid_points_2d(ax, pivot)

        if best_point is not None and balanced_point is not None:
            add_pins_to_heatmap(
                ax,
                pivot,
                best_point,
                balanced_point,
                metric,
                scenario_name,
                val_comparison_df,
                equal_point=equal_point,
            )

        x_labels = [str(x) for x in pivot.columns]
        y_labels = [str(y) for y in pivot.index]

        x_pos = np.arange(len(x_labels))
        y_pos = np.arange(len(y_labels))

        step = 2

        ax.set_xticks(x_pos[::step])
        ax.set_yticks(y_pos[::step])
        ax.set_xticklabels(x_labels[::step], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(y_labels[::step], fontsize=8)

        ax.set_xlabel("c1")
        ax.set_ylabel("c0")
        ax.set_title(format_metric_label_for_plot(metric, include_split=False), fontsize=11)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(format_metric_label_for_plot(metric, include_split=True), rotation=90, labelpad=10)

    fig.suptitle(title, fontsize=15)
    add_2d_grid_note(fig)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(get_scenario_figure_path(scenario_name, filename), dpi=180)
    plt.close(fig)


def save_validation_figures(agg_df, best_df, balanced_df, val_comparison_df=None):
    validation_mean_panels = [
        {
            "name": "val_main_mean",
            "title": "VALIDACIÓN · Mean · Métricas principales",
            "metrics": ["val_f1_mean", "val_pr_auc_mean", "val_balanced_accuracy_mean", "val_mcc_mean"],
        },
        {
            "name": "val_metrics_per_class_mean",
            "title": "VALIDACIÓN · Mean · Métricas por clase",
            "metrics": ["val_recall_mean", "val_fnr_mean", "val_specificity_mean", "val_roc_auc_mean"],
        },
    ]

    validation_std_panels = [
        {
            "name": "val_main_std",
            "title": "VALIDACIÓN · STD · Métricas principales",
            "metrics": ["val_f1_std", "val_pr_auc_std", "val_balanced_accuracy_std", "val_mcc_std"],
        },
        {
            "name": "val_metrics_per_class_std",
            "title": "VALIDACIÓN · STD · Métricas por clase",
            "metrics": ["val_recall_std", "val_fnr_std", "val_specificity_std", "val_roc_auc_std"],
        },
    ]

    for scenario_name in SCENARIOS.keys():
        best_point, balanced_point, equal_point = get_pin_points(best_df, balanced_df, scenario_name)

        # VALIDACIÓN MEAN: con pines.
        for panel in validation_mean_panels:
            save_heatmap_panel(
                agg_df,
                scenario_name,
                panel["metrics"],
                f"{format_scenario_label_for_plot(scenario_name)} · {panel['title']} ({len(SEEDS)} seeds)",
                f"{scenario_name}_panel_{panel['name']}_with_pins.png",
                best_point,
                balanced_point,
                val_comparison_df,
                equal_point=equal_point,
            )

            for metric in panel["metrics"]:
                save_3d_surface_plot(
                    agg_df,
                    scenario_name,
                    metric,
                    f"{scenario_name}_3d_{metric}_with_pins.png",
                    best_point,
                    balanced_point,
                    val_comparison_df,
                    equal_point=equal_point,
                )

        # VALIDACIÓN STD: sin pines.
        for panel in validation_std_panels:
            save_heatmap_panel(
                agg_df,
                scenario_name,
                panel["metrics"],
                f"{format_scenario_label_for_plot(scenario_name)} · {panel['title']} ({len(SEEDS)} seeds)",
                f"{scenario_name}_panel_{panel['name']}.png",
                best_point=None,
                balanced_point=None,
                val_comparison_df=None,
            )

            for metric in panel["metrics"]:
                save_3d_surface_plot(
                    agg_df,
                    scenario_name,
                    metric,
                    f"{scenario_name}_3d_{metric}.png",
                    best_point=None,
                    balanced_point=None,
                    val_comparison_df=None,
                )


# ============================================================
# 6. FIGURAS COMPARATIVAS DE VALIDACIÓN Y TEST
# ============================================================

def format_weight_for_bar(value):
    """
    Formatea pesos de clase para las etiquetas de las barras.
    Se usa notación compacta para que quepa dentro de la figura.
    """
    if pd.isna(value):
        return "nan"

    return f"{float(value):.3g}"


def get_model_plot_style(model_name):
    """
    Devuelve nombre corto y color fijo para cada modelo final.

    Mantener colores fijos ayuda a interpretar las figuras comparativas:
    - Manual / Best F1: naranja/vermillion suave.
    - Equal weights: violeta suave.
    - Sklearn balanced: azul profundo.
    """
    styles = {
        "best_val_f1_grid": {
            "label": "Manual (Best F1)",
            "short_label": "Manual",
            "color": MODEL_COLORS["best_val_f1_grid"],
        },
        "equal_weights": {
            "label": "Equal weights",
            "short_label": "Equal weights",
            "color": MODEL_COLORS["equal_weights"],
        },
        "sklearn_balanced": {
            "label": "Sklearn balanced",
            "short_label": "Sklearn balanced",
            "color": MODEL_COLORS["sklearn_balanced"],
        },
    }

    return styles.get(
        model_name,
        {
            "label": model_name,
            "short_label": model_name,
            "color": "#7f7f7f",
        },
    )


def save_final_comparison_plots(final_agg_df, split_name):
    """
    Guarda figuras comparativas finales en validación o test.

    - Barras agrupadas por escenario.
    - Color fijo por modelo.
    - Etiquetas con valor de la métrica y pesos c0/c1.
    - Barras de error con la desviación típica de la métrica entre semillas.

    """
    metrics_to_plot = [
        f"{split_name}_f1_mean",
        f"{split_name}_pr_auc_mean",
        f"{split_name}_balanced_accuracy_mean",
        f"{split_name}_mcc_mean",
        f"{split_name}_recall_mean",
        f"{split_name}_fnr_mean",
        f"{split_name}_specificity_mean",
        f"{split_name}_roc_auc_mean",
    ]

    model_order = [
        "best_val_f1_grid",
        "equal_weights",
        "sklearn_balanced",
    ]

    scenario_order = [
        scenario_name
        for scenario_name in SCENARIOS.keys()
        if scenario_name in set(final_agg_df["scenario"].unique())
    ]

    for metric in metrics_to_plot:
        if metric not in final_agg_df.columns:
            continue

        metric_std_col = metric.replace("_mean", "_std")

        available_models = [
            model_name
            for model_name in model_order
            if model_name in set(final_agg_df["model_name"].unique())
        ]

        if not available_models or not scenario_order:
            continue

        x = np.arange(len(scenario_order))
        bar_width = 0.24 if len(available_models) >= 3 else 0.32

        fig, ax = plt.subplots(figsize=(14.2, 7.0))

        all_values_for_limits = []

        for model_idx, model_name in enumerate(available_models):
            style = get_model_plot_style(model_name)
            offset = (model_idx - (len(available_models) - 1) / 2.0) * bar_width

            values = []
            std_values = []
            c0_values = []
            c1_values = []

            for scenario_name in scenario_order:
                row = final_agg_df[
                    (final_agg_df["scenario"] == scenario_name)
                    & (final_agg_df["model_name"] == model_name)
                ]

                if row.empty:
                    values.append(np.nan)
                    std_values.append(0.0)
                    c0_values.append(np.nan)
                    c1_values.append(np.nan)
                    continue

                row = row.iloc[0]
                values.append(float(row[metric]))

                if metric_std_col in final_agg_df.columns and not pd.isna(row[metric_std_col]):
                    std_values.append(float(row[metric_std_col]))
                else:
                    std_values.append(0.0)

                c0_values.append(float(row["c0_mean"]) if "c0_mean" in final_agg_df.columns else np.nan)
                c1_values.append(float(row["c1_mean"]) if "c1_mean" in final_agg_df.columns else np.nan)

            values_array = np.asarray(values, dtype=float)
            std_array = np.asarray(std_values, dtype=float)
            all_values_for_limits.extend(values_array[~np.isnan(values_array)].tolist())

            bars = ax.bar(
                x + offset,
                values_array,
                width=bar_width,
                label=style["label"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.8,
                yerr=std_array if metric_std_col in final_agg_df.columns else None,
                error_kw={
                    "elinewidth": 1.0,
                    "capsize": 3,
                    "capthick": 1.0,
                    "ecolor": "#333333",
                    "alpha": 0.85,
                },
            )

            for bar, value, std_value, c0_value, c1_value in zip(
                bars,
                values_array,
                std_array,
                c0_values,
                c1_values,
            ):
                if pd.isna(value):
                    continue

                label_text = (
                    f"mean={value:.3f}\n"
                    f"std={std_value:.3f}\n"
                    f"c0={format_weight_for_bar(c0_value)}\n"
                    f"c1={format_weight_for_bar(c1_value)}"
                )

                bar_center_x = bar.get_x() + bar.get_width() / 2.0

                # En barras suficientemente altas, el texto va dentro.
                # En barras bajas, el texto va fuera para que no quede ilegible.
                if value >= 0.28:
                    text_y = value - 0.035
                    vertical_alignment = "top"
                    text_color = "white"
                    bbox = dict(
                        boxstyle="round,pad=0.16",
                        fc="black",
                        ec="none",
                        alpha=0.24,
                    )
                else:
                    text_y = value + max(0.035, std_value + 0.020)
                    vertical_alignment = "bottom"
                    text_color = "#222222"
                    bbox = dict(
                        boxstyle="round,pad=0.16",
                        fc="white",
                        ec=style["color"],
                        alpha=0.92,
                        linewidth=0.6,
                    )

                ax.text(
                    bar_center_x,
                    text_y,
                    label_text,
                    ha="center",
                    va=vertical_alignment,
                    fontsize=7.1,
                    color=text_color,
                    rotation=0,
                    bbox=bbox,
                    zorder=30,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([format_scenario_label_for_plot(scenario_name) for scenario_name in scenario_order], fontsize=10)
        ax.set_xlabel("Escenario")
        ax.set_ylabel(format_metric_axis_label_for_plot(metric))
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name.upper())
        ax.set_title(
            f"Comparación final en {split_display} · {format_metric_label_for_plot(metric, include_split=False)}",
            fontsize=14,
            pad=12,
        )

        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        for boundary in np.arange(0.5, len(scenario_order) - 0.5, 1.0):
            ax.axvline(boundary, color="#dddddd", linewidth=0.8, linestyle="--", alpha=0.7)

        if all_values_for_limits:
            min_value = min(all_values_for_limits)
            max_value = max(all_values_for_limits)
        else:
            min_value = 0.0
            max_value = 1.0

        lower_limit = min(0.0, min_value - 0.10)
        upper_limit = max(1.0, max_value + 0.20)

        if metric.endswith("_mcc_mean"):
            lower_limit = min(-0.05, min_value - 0.12)
            upper_limit = max(0.20, max_value + 0.20)

        ax.set_ylim(lower_limit, upper_limit)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=len(available_models),
            frameon=True,
            fontsize=9,
        )

        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(get_global_figure_path(f"{split_name}_final_comparison_{metric}.png"), dpi=180)
        plt.close(fig)



def _draw_confusion_matrix_on_axis(ax, cm, title):
    """
    Dibuja una matriz de confusión 2x2 en un eje.

    La intensidad de color se normaliza por fila real para que se vea bien
    también en escenarios desbalanceados. El texto muestra:
    - conteo absoluto;
    - porcentaje dentro de cada clase real.
    """
    cm = np.asarray(cm, dtype=int)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row_pct = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    im = ax.imshow(cm_row_pct, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
    ax.set_yticklabels(["Real 0", "Real 1"], fontsize=9)
    ax.set_xlabel("Predicción", fontsize=9)
    ax.set_ylabel("Clase real", fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)

    for i in range(2):
        for j in range(2):
            pct = cm_row_pct[i, j] * 100.0
            text_color = "white" if cm_row_pct[i, j] >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def save_final_confusion_matrix_panels(predictions_path, split_name):
    """
    Guarda paneles globales de matrices de confusión para VALIDACIÓN o TEST.

    Se genera un panel 2x2 por estrategia:
    - Best F1.
    - Equal weights.
    - Sklearn balanced.

    Cada panel contiene los cuatro escenarios sintéticos:
    ideal, intermedio, avanzado y dificil.

    No modifica entrenamiento, evaluación, métricas ni predicciones. Solo lee
    el CSV de predicciones finales ya calculado y representa y_true vs y_pred.
    """
    predictions_path = Path(predictions_path)

    if not predictions_path.exists():
        return

    predictions_df = pd.read_csv(predictions_path)

    if predictions_df.empty:
        return

    required_cols = {"scenario", "model_name", "y_true", "y_pred"}
    if not required_cols.issubset(set(predictions_df.columns)):
        return

    if "split" in predictions_df.columns:
        predictions_df = predictions_df[predictions_df["split"] == split_name].copy()

    if predictions_df.empty:
        return

    model_order = ["best_val_f1_grid", "equal_weights", "sklearn_balanced"]
    available_models = [
        model_name
        for model_name in model_order
        if model_name in set(predictions_df["model_name"].unique())
    ]

    scenario_order = [
        scenario_name
        for scenario_name in SCENARIOS.keys()
        if scenario_name in set(predictions_df["scenario"].unique())
    ]

    if not available_models or not scenario_order:
        return

    for model_name in available_models:
        style = get_model_plot_style(model_name)
        model_predictions_df = predictions_df[predictions_df["model_name"] == model_name].copy()

        fig, axes = plt.subplots(2, 2, figsize=(16.4, 12.2))
        axes = axes.ravel()

        last_im = None

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(scenario_order):
                ax.axis("off")
                continue

            scenario_name = scenario_order[ax_idx]
            group = model_predictions_df[
                model_predictions_df["scenario"] == scenario_name
            ].copy()

            if group.empty:
                ax.axis("off")
                continue

            y_true = group["y_true"].to_numpy(dtype=int)
            y_pred = group["y_pred"].to_numpy(dtype=int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            cm = np.array([[tn, fp], [fn, tp]], dtype=int)

            n_total = int(len(group))
            n_pos = int(np.sum(y_true == 1))

            if "seed" in group.columns:
                n_seeds_group = int(group["seed"].nunique())
                title = (
                    f"{format_scenario_label_for_plot(scenario_name)}\n"
                    f"{SPLIT_DISPLAY_NAMES.get(split_name, split_name.upper())} · agregada {n_seeds_group} seeds\n"
                    f"n total={n_total}, positivos={n_pos}"
                )
            else:
                title = (
                    f"{format_scenario_label_for_plot(scenario_name)}\n"
                    f"{SPLIT_DISPLAY_NAMES.get(split_name, split_name.upper())} · agregada\n"
                    f"n total={n_total}, positivos={n_pos}"
                )

            last_im = _draw_confusion_matrix_on_axis(ax, cm, title)

        if "seed" in model_predictions_df.columns:
            n_seeds_panel = int(model_predictions_df["seed"].nunique())
            aggregation_text = f"agregadas entre {n_seeds_panel} seeds"
            footer_text = (
                f"Lectura: cada matriz acumula las predicciones finales de {str(split_name).upper()} "
                f"obtenidas en {n_seeds_panel} seeds para cada escenario. "
                "El n mostrado es el total agregado de predicciones, no el tamaño de una única seed. "
                "Cada celda muestra conteo acumulado y porcentaje por fila real."
            )
        else:
            aggregation_text = "agregadas"
            footer_text = (
                f"Lectura: cada matriz acumula las predicciones finales de {str(split_name).upper()}. "
                "El n mostrado es el total agregado de predicciones. "
                "Cada celda muestra conteo acumulado y porcentaje por fila real."
            )

        fig.suptitle(
            f"{SPLIT_DISPLAY_NAMES.get(split_name, split_name.upper())} FINAL · Matrices de confusión {aggregation_text} · {style['label']}",
            fontsize=19,
            y=0.979,
        )

        fig.text(
            0.5,
            0.035,
            footer_text,
            ha="center",
            va="center",
            fontsize=8.7,
            color="#444444",
            wrap=True,
        )

        # Ajuste manual del panel para evitar warnings de tight_layout y para
        # que la barra de color no se superponga a las matrices de la derecha.
        fig.subplots_adjust(
            left=0.060,
            right=0.895,
            bottom=0.105,
            top=0.895,
            wspace=0.08,
            hspace=0.26,
        )

        if last_im is not None:
            cbar_ax = fig.add_axes([0.930, 0.205, 0.018, 0.545])
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label("Porcentaje dentro de cada clase real", rotation=90, labelpad=12)

        fig.savefig(
            get_global_figure_path(f"{split_name}_confusion_matrix_panel_{model_name}.png"),
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.12,
        )
        plt.close(fig)


def save_test_final_comparison_plots(test_agg_df):
    save_final_comparison_plots(test_agg_df, split_name="test")


def save_val_final_comparison_plots(val_agg_df):
    save_final_comparison_plots(val_agg_df, split_name="val")



def save_final_boxplot_plots_and_tables(final_results_df, split_name):
    """
    Guarda figuras de caja y bigotes para la comparación final en validación o test.

    Esta función no modifica el entrenamiento ni la evaluación. Solo utiliza los
    resultados finales ya calculados para representar la distribución de cada
    métrica a lo largo de las semillas definidas en SEEDS.

    Lectura de la figura:
    - La caja representa el rango intercuartílico Q1-Q3.
    - La línea negra central representa la mediana.
    - El círculo blanco grande representa la media.
    - Los bigotes representan el rango de valores no atípicos.
    - Los círculos pequeños fuera de los bigotes representan valores atípicos.
    """
    metrics_to_plot = [
        f"{split_name}_f1",
        f"{split_name}_pr_auc",
        f"{split_name}_balanced_accuracy",
        f"{split_name}_mcc",
        f"{split_name}_recall",
        f"{split_name}_fnr",
        f"{split_name}_specificity",
        f"{split_name}_roc_auc",
    ]

    model_order = [
        "best_val_f1_grid",
        "equal_weights",
        "sklearn_balanced",
    ]

    scenario_order = [
        scenario_name
        for scenario_name in SCENARIOS.keys()
        if scenario_name in set(final_results_df["scenario"].unique())
    ]

    summary_rows = []

    for metric in metrics_to_plot:
        if metric not in final_results_df.columns:
            continue

        available_models = [
            model_name
            for model_name in model_order
            if model_name in set(final_results_df["model_name"].unique())
        ]

        if not available_models or not scenario_order:
            continue

        fig, ax = plt.subplots(figsize=(14.8, 7.8))

        box_data = []
        box_positions = []
        box_colors = []

        group_width = len(available_models) + 1
        xtick_positions = []
        xtick_labels = []

        for scenario_idx, scenario_name in enumerate(scenario_order):
            base_position = scenario_idx * group_width
            xtick_positions.append(base_position + (len(available_models) - 1) / 2.0)
            xtick_labels.append(format_scenario_label_for_plot(scenario_name))

            for model_idx, model_name in enumerate(available_models):
                values = (
                    final_results_df[
                        (final_results_df["scenario"] == scenario_name)
                        & (final_results_df["model_name"] == model_name)
                    ][metric]
                    .dropna()
                    .to_numpy(dtype=float)
                )

                if len(values) == 0:
                    continue

                style = get_model_plot_style(model_name)
                position = base_position + model_idx

                box_data.append(values)
                box_positions.append(position)
                box_colors.append(style["color"])

                q1 = float(np.percentile(values, 25))
                q3 = float(np.percentile(values, 75))

                summary_rows.append({
                    "split": split_name,
                    "metric": metric,
                    "scenario": scenario_name,
                    "model_name": model_name,
                    "n_seeds": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                })

        if not box_data:
            plt.close(fig)
            continue

        boxplot = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.58,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#111111",
                markeredgewidth=1.05,
                markersize=5.2,
            ),
            flierprops=dict(
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#111111",
                markeredgewidth=0.9,
                markersize=3.8,
                linestyle="none",
                alpha=0.95,
            ),
        )

        # Se usa exactamente la misma paleta MODEL_COLORS que en las barras,
        # mapas y superficies. No se aplica transparencia al relleno para que
        # el color del modelo sea coherente entre todas las figuras.
        for patch, color in zip(boxplot["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(1.00)
            patch.set_edgecolor("#222222")
            patch.set_linewidth(0.9)

        for median in boxplot["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.25)

        for whisker in boxplot["whiskers"]:
            whisker.set_color("#333333")
            whisker.set_linewidth(0.9)

        for cap in boxplot["caps"]:
            cap.set_color("#333333")
            cap.set_linewidth(0.9)

        for mean_marker in boxplot["means"]:
            mean_marker.set_marker("o")
            mean_marker.set_markerfacecolor("white")
            mean_marker.set_markeredgecolor("#111111")
            mean_marker.set_markersize(5.2)

        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=10)
        ax.set_xlabel("Escenario")
        ax.set_ylabel(format_metric_axis_label_for_plot(metric))
        split_display = SPLIT_DISPLAY_NAMES.get(split_name, split_name.upper())
        ax.set_title(
            f"Caja y bigotes en {split_display} · {format_metric_label_for_plot(metric, include_split=False)}",
            fontsize=14,
            pad=12,
        )

        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        for scenario_idx in range(1, len(scenario_order)):
            boundary = scenario_idx * group_width - 0.5
            ax.axvline(boundary, color="#dddddd", linewidth=0.8, linestyle="--", alpha=0.7)

        model_handles = []
        for model_name in available_models:
            style = get_model_plot_style(model_name)
            model_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linewidth=6.0,
                    solid_capstyle="butt",
                    label=style["label"],
                )
            )

        boxplot_reading_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor="#222222",
                label="Caja: Q1-Q3 (50% central)",
            ),
            Line2D(
                [0],
                [0],
                color="#111111",
                linewidth=1.4,
                label="Línea negra: mediana",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor="#111111",
                label="Círculo blanco: media",
            ),
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=1.0,
                label="Bigotes: rango no atípico",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=3.8,
                markerfacecolor="white",
                markeredgecolor="#111111",
                label="Puntos fuera: valores atípicos",
            ),
        ]

        # Leyenda de lectura colocada fuera del área de datos para evitar
        # que tape cajas, bigotes, medias u outliers cuando las métricas quedan
        # muy cerca del borde superior del gráfico.
        fig.legend(
            handles=boxplot_reading_handles,
            loc="center right",
            bbox_to_anchor=(0.985, 0.56),
            frameon=True,
            framealpha=0.96,
            fontsize=8.4,
            title="Lectura caja-bigotes",
            title_fontsize=9.2,
            borderpad=0.55,
            labelspacing=0.42,
            handlelength=1.6,
            handletextpad=0.6,
        )

        # Leyenda inferior de modelos, igual que en las figuras comparativas:
        # situada fuera del gráfico para no solaparse con las cajas y usando
        # exactamente los colores definidos en MODEL_COLORS.
        fig.legend(
            handles=model_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.085),
            ncol=len(available_models),
            frameon=True,
            fontsize=9.5,
            borderpad=0.45,
            labelspacing=0.45,
            handlelength=1.6,
            handletextpad=0.7,
            columnspacing=1.6,
        )

        # Se reserva margen derecho para la leyenda explicativa y margen inferior
        # para la leyenda de modelos. Así ninguna leyenda se superpone con las
        # cajas y bigotes.
        fig.tight_layout(rect=[0, 0.16, 0.80, 1])
        fig.savefig(get_global_figure_path(f"{split_name}_final_boxplot_{metric}.png"), dpi=180, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / f"{split_name}_final_boxplot_summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df






# ============================================================
# 7. COMPROBACIÓN DE RESULTADOS EXISTENTES
# ============================================================

def write_experiment_metadata():
    metadata = {
        "run_signature": RUN_SIGNATURE,
        "model": "LogisticRegression estándar de Scikit-Learn",
        "logistic_regression_config": {
            "class_weight": "variable según c0/c1",
            "random_state": "seed",
            "default_solver": "lbfgs",
            "default_penalty": "l2",
            "max_iter": MODEL_MAX_ITER,
        },
        "c_values": C_VALUES,
        "seeds": SEEDS,
        "grid_size": len(C_VALUES) * len(C_VALUES),
        "grid_uses_train_metrics": False,
        "grid_saves_validation_predictions": True,
        "convergence_logging": True,
        "convergence_columns": [
            "n_iter",
            "max_iter",
            "convergence_warning",
            "reached_max_iter",
            "converged_without_warning",
            "convergence_message",
        ],
        "output_dir": str(OUTPUT_DIR.resolve()),
    }

    with open(EXPERIMENT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def read_experiment_metadata():
    if not EXPERIMENT_METADATA_PATH.exists():
        return None

    try:
        with open(EXPERIMENT_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def existing_val_grid_outputs_are_compatible():
    """
    Comprueba si ya existen resultados compatibles con esta versión.

    Para saltar el entrenamiento de la rejilla no basta con tener solo
    coeficientes: también hacen falta métricas de validación para las superficies
    y predicciones de validación si quieres curvas/errores sin recalcular.
    """
    metadata = read_experiment_metadata()

    if metadata is None:
        return False

    if metadata.get("run_signature") != RUN_SIGNATURE:
        return False

    if metadata.get("c_values") != C_VALUES:
        return False

    if metadata.get("seeds") != SEEDS:
        return False

    required_paths = [
        RAW_RESULTS_PATH,
        RAW_PREDICTIONS_PATH,
        AGG_RESULTS_PATH,
        BEST_CONFIGS_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            return False

    try:
        temp_df = pd.read_csv(RAW_RESULTS_PATH, nrows=5)
        scenarios_df = pd.read_csv(RAW_RESULTS_PATH, usecols=["scenario"])
        pred_df = pd.read_csv(RAW_PREDICTIONS_PATH, nrows=5)
    except Exception:
        return False

    required_cols = {"scenario", "seed", "c0", "c1", "val_f1", "n_iter", "convergence_warning"}
    if not required_cols.issubset(set(temp_df.columns)):
        return False

    # La rejilla nueva solo guarda métricas de validación.
    has_train_grid_columns = any(col.startswith("train_") for col in temp_df.columns)
    if has_train_grid_columns:
        return False

    required_pred_cols = {"scenario", "seed", "model_name", "split", "sample_id", "c0", "c1", "y_true", "y_pred", "y_prob"}
    if not required_pred_cols.issubset(set(pred_df.columns)):
        return False

    existing_scenarios = set(scenarios_df["scenario"].unique().tolist())
    expected_scenarios = set(SCENARIOS.keys())

    if existing_scenarios != expected_scenarios:
        return False

    coefficients_wide_ready = all(
        get_coefficients_wide_path(scenario_name, "grid_manual").exists()
        for scenario_name in SCENARIOS.keys()
    )

    if not coefficients_wide_ready:
        return False

    return True


def remove_old_incompatible_outputs():
    paths_to_remove = [
        RAW_RESULTS_PATH,
        RAW_PREDICTIONS_PATH,
        AGG_RESULTS_PATH,
        BEST_CONFIGS_PATH,
        BALANCED_POINTS_PATH,
        BALANCED_POINTS_RAW_PATH,
        VAL_FINAL_RAW_RESULTS_PATH,
        VAL_FINAL_RAW_PREDICTIONS_PATH,
        VAL_FINAL_AGG_RESULTS_PATH,
        VAL_FINAL_COMPARISON_PATH,
        TEST_FINAL_RAW_RESULTS_PATH,
        TEST_FINAL_RAW_PREDICTIONS_PATH,
        TEST_FINAL_AGG_RESULTS_PATH,
        TEST_FINAL_COMPARISON_PATH,
        TIMERS_PATH,
        EXPERIMENT_METADATA_PATH,
    ]

    for path in paths_to_remove:
        if path.exists():
            path.unlink()

    for scenario_name in SCENARIOS.keys():
        for strategy_name in ["grid_manual", "equal_weights", "sklearn_balanced"]:
            coeff_path = get_coefficients_wide_path(scenario_name, strategy_name)
            if coeff_path.exists():
                coeff_path.unlink()

        # Limpieza de posibles archivos antiguos que estaban directamente en coefficients_wide.
        old_coeff_path = COEFFICIENTS_WIDE_DIR / f"{scenario_name}_coefficients_wide.csv"
        if old_coeff_path.exists():
            old_coeff_path.unlink()


def raw_final_outputs_have_expected_scenarios(path):
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path, usecols=["scenario", "model_name"])
    except Exception:
        return False

    expected_scenarios = set(SCENARIOS.keys())
    existing_scenarios = set(df["scenario"].unique().tolist())

    expected_models = FINAL_MODEL_NAMES
    existing_models = set(df["model_name"].unique().tolist())

    return existing_scenarios == expected_scenarios and existing_models == expected_models


# ============================================================
# 8. TABLAS COMPARATIVAS FINALES
# ============================================================

def aggregate_final_results(final_results_df, group_cols):
    metric_cols = [
        col for col in final_results_df.columns
        if col not in ["scenario", "seed", "model_name", "convergence_message"]
        and pd.api.types.is_numeric_dtype(final_results_df[col])
    ]

    agg_df = (
        final_results_df
        .groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    new_columns = []

    for col in agg_df.columns:
        if col[1] == "":
            new_columns.append(col[0])
        else:
            new_columns.append(f"{col[0]}_{col[1]}")

    agg_df.columns = new_columns

    std_cols = [col for col in agg_df.columns if col.endswith("_std")]
    agg_df[std_cols] = agg_df[std_cols].fillna(0.0)

    agg_df = agg_df.sort_values(group_cols).reset_index(drop=True)

    return agg_df


def make_comparison_table(final_agg_df, split_name):
    cols = [
        "scenario",
        "model_name",
        "c0_mean",
        "c0_std",
        "c1_mean",
        "c1_std",

        f"{split_name}_f1_mean",
        f"{split_name}_f1_std",

        f"{split_name}_pr_auc_mean",
        f"{split_name}_pr_auc_std",

        f"{split_name}_balanced_accuracy_mean",
        f"{split_name}_balanced_accuracy_std",

        f"{split_name}_mcc_mean",
        f"{split_name}_mcc_std",

        f"{split_name}_recall_mean",
        f"{split_name}_recall_std",

        f"{split_name}_fnr_mean",
        f"{split_name}_fnr_std",

        f"{split_name}_specificity_mean",
        f"{split_name}_specificity_std",

        f"{split_name}_roc_auc_mean",
        f"{split_name}_roc_auc_std",

    ]

    existing_cols = [col for col in cols if col in final_agg_df.columns]
    return final_agg_df[existing_cols].copy()


def print_comparison_table_by_scenario(df, title, split_name, decimals=4):
    """
    Imprime la tabla comparativa completa en terminal, pero dividida por escenario
    y por bloques de columnas para que no salga una línea gigante ilegible.
    """
    df_print = df.copy().round(decimals)

    blocks = [
        {
            "title": "Pesos de clase",
            "cols": [
                "scenario",
                "model_name",
                "c0_mean",
                "c0_std",
                "c1_mean",
                "c1_std",
            ],
        },
        {
            "title": "Métricas principales",
            "cols": [
                "scenario",
                "model_name",
                f"{split_name}_f1_mean",
                f"{split_name}_f1_std",
                f"{split_name}_pr_auc_mean",
                f"{split_name}_pr_auc_std",
                f"{split_name}_balanced_accuracy_mean",
                f"{split_name}_balanced_accuracy_std",
                f"{split_name}_mcc_mean",
                f"{split_name}_mcc_std",
            ],
        },
        {
            "title": "Métricas por clase y curvas",
            "cols": [
                "scenario",
                "model_name",
                f"{split_name}_recall_mean",
                f"{split_name}_recall_std",
                f"{split_name}_fnr_mean",
                f"{split_name}_fnr_std",
                f"{split_name}_specificity_mean",
                f"{split_name}_specificity_std",
                f"{split_name}_roc_auc_mean",
                f"{split_name}_roc_auc_std",
            ],
        },
    ]

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    for scenario_name in SCENARIOS.keys():
        temp_scenario = df_print[df_print["scenario"] == scenario_name].copy()

        if temp_scenario.empty:
            continue

        print("\n" + "#" * 110)
        print(f"ESCENARIO: {scenario_name.upper()}")
        print("#" * 110)

        for block in blocks:
            existing_cols = [
                col for col in block["cols"]
                if col in temp_scenario.columns
            ]

            if len(existing_cols) <= 2:
                continue

            print("\n" + "-" * 110)
            print(block["title"])
            print("-" * 110)
            print(temp_scenario[existing_cols].to_string(index=False))


# ============================================================
# 9. BUCLE PRINCIPAL
# ============================================================

all_results_list = []
all_predictions_list = []

def save_timers(timer_frames):
    """
    Guarda únicamente los tiempos reales útiles de ejecución.

    Se eliminan del archivo de tiempos los tiempos internos acumulados por modelo/seed porque
    no representan el tiempo real observado en pantalla y pueden confundir el
    análisis de ahorro computacional.

    El TXT timers_execution.txt queda centrado en:
    - tiempos wall-clock de los bloques principales;
    - número de modelos del bloque, cuando aplica;
    - tiempo real total del bloque.
    """
    frames = [df for df in timer_frames if df is not None and not df.empty]

    if not frames:
        return pd.DataFrame()

    timers_df = pd.concat(frames, ignore_index=True)

    useful_wall_stages = [
        "load_existing_grid_outputs",
        "load_existing_final_strategy_coefficients",
        "load_existing_final_test_outputs",

        "grid_manual_train_scenario_parallel_total",
        "equal_weights_train_scenario_parallel_total",
        "sklearn_balanced_train_scenario_parallel_total",

        "grid_manual_validation_scenario_parallel_total",
        "grid_aggregation_by_scenario",
        "grid_best_selection_by_scenario",
        "final_test_best_val_f1_grid_scenario_parallel_total",
        "final_test_equal_weights_scenario_parallel_total",
        "final_test_sklearn_balanced_scenario_parallel_total",

        "figures_total",
        "script_total",
    ]

    if "stage" in timers_df.columns and "model_name" in timers_df.columns:
        timers_df = timers_df[
            (timers_df["model_name"].astype(str) == "wall_clock")
            & (timers_df["stage"].astype(str).isin(useful_wall_stages))
        ].copy()

    useful_columns = [
        "scenario",
        "seed",
        "stage",
        "model_name",
        "n_models",
        "total_seconds",
        "total_time_readable",
        "skipped_because_existing_outputs",
    ]

    existing_columns = [col for col in useful_columns if col in timers_df.columns]
    timers_df = timers_df[existing_columns].copy()

    return timers_df



def add_wall_clock_timer(timer_frames, scenario_name, stage, start_time, n_models=None, skipped=False):
    total_seconds = time.perf_counter() - start_time
    timer_frames.append(pd.DataFrame([{
        "scenario": scenario_name,
        "seed": "all",
        "stage": stage,
        "model_name": "wall_clock",
        "n_models": n_models if n_models is not None else np.nan,
        "scaler_seconds": np.nan,
        "fit_seconds": np.nan,
        "validation_prediction_seconds": np.nan,
        "test_prediction_seconds": np.nan,
        "metrics_seconds": np.nan,
        "total_seconds": total_seconds,
        "total_time_readable": format_seconds(total_seconds),
        "used_saved_coefficients": np.nan,
        "skipped_because_existing_outputs": skipped,
    }]))



def print_clean_timer_summary(timers_df):
    """
    Resumen de tiempos reales comparables de los datasets sintéticos.

    Solo se muestran tiempos wall-clock. El objetivo es comparar el coste real
    hasta TEST de cada estrategia, dejando aparte la validación final informativa
    y la generación de figuras.
    """
    if timers_df is None or timers_df.empty:
        return

    timers_df = timers_df.copy()

    if "total_seconds" in timers_df.columns:
        timers_df["total_seconds"] = pd.to_numeric(
            timers_df["total_seconds"],
            errors="coerce",
        )

    if "n_models" in timers_df.columns:
        timers_df["n_models"] = pd.to_numeric(
            timers_df["n_models"],
            errors="coerce",
        )

    scenario_order = [s for s in SCENARIOS.keys()]
    grid_size = len(C_VALUES) * len(C_VALUES)
    n_seeds = len(SEEDS)

    def format_or_dash(value):
        if pd.isna(value):
            return "-"
        return format_seconds(value)

    def get_wall_rows(stage_name, scenario_name=None):
        rows = timers_df[timers_df["stage"].astype(str) == stage_name].copy()

        if "model_name" in rows.columns:
            rows = rows[rows["model_name"].astype(str) == "wall_clock"]

        if scenario_name is not None:
            rows = rows[rows["scenario"].astype(str) == str(scenario_name)]

        return rows

    def get_wall_seconds(stage_name, scenario_name=None):
        rows = get_wall_rows(stage_name, scenario_name)
        if rows.empty:
            return np.nan
        return float(rows["total_seconds"].sum())

    def get_wall_n_models(stage_name, scenario_name=None, fallback=None):
        rows = get_wall_rows(stage_name, scenario_name)

        if rows.empty or "n_models" not in rows.columns:
            if fallback is not None and not pd.isna(fallback):
                return int(fallback)
            return 0

        value = rows["n_models"].fillna(0.0).sum()

        if value == 0 and fallback is not None and not pd.isna(fallback):
            return int(fallback)

        return int(value)

    def safe_sum(values):
        total = 0.0
        any_value = False

        for value in values:
            if not pd.isna(value):
                total += float(value)
                any_value = True

        return total if any_value else np.nan

    print("\n" + "=" * 128)
    print("RESUMEN DE COSTE COMPUTACIONAL COMPARABLE · DATASETS SINTÉTICOS")
    print("=" * 128)

    print("Lectura rápida:")
    print("- Solo se muestran tiempos reales / wall-clock.")
    print("- La tabla principal sirve para comparar después con SMOTE, sobregeneración aleatoria, subgeneración aleatoria o una heurística.")
    print("- Preparación/remuestreo: ahora no aplica; en SMOTE/oversampling/undersampling será el tiempo de generar el train remuestreado.")
    print("- Entrenamiento: tiempo de ajustar los modelos en TRAIN.")
    print("- Validación rejilla: solo aplica a Weighted grid; evalúa todos los pares (c0, c1) en VALIDACIÓN sin reentrenar.")
    print("- Agregación/selección: solo aplica a Weighted grid; calcula mean/std entre seeds y elige el mejor par por val_f1_mean.")
    print("- TEST final: evaluación final sobre TEST, sin volver a entrenar ni seleccionar.")
    print("- La validación final comparativa de estrategias se sigue guardando, pero no se suma al coste comparable porque es informativa.")
    print("- El script ejecuta los bloques de forma secuencial: termina uno y después pasa al siguiente.")
    print(f"- Dentro de cada bloque se paralelizan las {n_seeds} seeds con joblib; no se paralelizan todos los escenarios a la vez.")
    print(f"- Rejilla weighted actual: {len(C_VALUES)} x {len(C_VALUES)} = {grid_size} combinaciones de pesos por seed x {n_seeds} seeds.")
    print("- Las figuras se guardan aparte y tampoco forman parte del coste experimental comparable hasta TEST.")

    # ============================================================
    # 0.0) COSTE COMPUTACIONAL COMPARABLE HASTA TEST
    # ============================================================
    comparable_rows = []

    for scenario_name in scenario_order:
        grid_train = get_wall_seconds("grid_manual_train_scenario_parallel_total", scenario_name)
        grid_val = get_wall_seconds("grid_manual_validation_scenario_parallel_total", scenario_name)
        grid_agg = get_wall_seconds("grid_aggregation_by_scenario", scenario_name)
        grid_selection = get_wall_seconds("grid_best_selection_by_scenario", scenario_name)
        best_test = get_wall_seconds("final_test_best_val_f1_grid_scenario_parallel_total", scenario_name)

        equal_train = get_wall_seconds("equal_weights_train_scenario_parallel_total", scenario_name)
        equal_test = get_wall_seconds("final_test_equal_weights_scenario_parallel_total", scenario_name)

        sklearn_train = get_wall_seconds("sklearn_balanced_train_scenario_parallel_total", scenario_name)
        sklearn_test = get_wall_seconds("final_test_sklearn_balanced_scenario_parallel_total", scenario_name)

        comparable_rows.append({
            "escenario": scenario_name,
            "estrategia": "Weighted grid / Best F1",
            "modelos entrenados TRAIN": get_wall_n_models(
                "grid_manual_train_scenario_parallel_total",
                scenario_name,
                fallback=get_total_grid_train_models(),
            ),
            "preparación/remuestreo": "No aplica",
            "entrenamiento": format_or_dash(grid_train),
            "validación rejilla": format_or_dash(grid_val),
            "agregación/selección": format_or_dash(safe_sum([grid_agg, grid_selection])),
            "test final": format_or_dash(best_test),
            "total comparable": format_or_dash(safe_sum([grid_train, grid_val, grid_agg, grid_selection, best_test])),
        })

        comparable_rows.append({
            "escenario": scenario_name,
            "estrategia": "Equal weights",
            "modelos entrenados TRAIN": get_wall_n_models(
                "equal_weights_train_scenario_parallel_total",
                scenario_name,
                fallback=get_total_single_strategy_train_models(),
            ),
            "preparación/remuestreo": "No aplica",
            "entrenamiento": format_or_dash(equal_train),
            "validación rejilla": "No aplica",
            "agregación/selección": "No aplica",
            "test final": format_or_dash(equal_test),
            "total comparable": format_or_dash(safe_sum([equal_train, equal_test])),
        })

        comparable_rows.append({
            "escenario": scenario_name,
            "estrategia": "Sklearn balanced",
            "modelos entrenados TRAIN": get_wall_n_models(
                "sklearn_balanced_train_scenario_parallel_total",
                scenario_name,
                fallback=get_total_single_strategy_train_models(),
            ),
            "preparación/remuestreo": "No aplica",
            "entrenamiento": format_or_dash(sklearn_train),
            "validación rejilla": "No aplica",
            "agregación/selección": "No aplica",
            "test final": format_or_dash(sklearn_test),
            "total comparable": format_or_dash(safe_sum([sklearn_train, sklearn_test])),
        })

    print("\n0.0) COSTE COMPUTACIONAL COMPARABLE HASTA TEST")
    print("   Tabla principal: preparación/remuestreo + entrenamiento + validación de rejilla + agregación/selección + TEST final.")
    print("   Weighted grid no remuestrea; su búsqueda consiste en entrenar la rejilla, validarla, agregar las seeds y seleccionar Best F1.")
    print("   Equal weights y Sklearn balanced no tienen búsqueda: solo entrenan su modelo y se evalúan en TEST.")
    print(pd.DataFrame(comparable_rows).to_string(index=False))

    # ============================================================
    # 0.1) DESGLOSE DEL ENTRENAMIENTO
    # ============================================================
    train_wall_rows = []

    for scenario_name in scenario_order:
        grid_train_wall = get_wall_seconds("grid_manual_train_scenario_parallel_total", scenario_name)
        equal_train_wall = get_wall_seconds("equal_weights_train_scenario_parallel_total", scenario_name)
        sklearn_train_wall = get_wall_seconds("sklearn_balanced_train_scenario_parallel_total", scenario_name)
        total_train_wall = safe_sum([grid_train_wall, equal_train_wall, sklearn_train_wall])

        train_wall_rows.append({
            "escenario": scenario_name,
            "TRAIN rejilla": format_or_dash(grid_train_wall),
            "modelos rejilla": get_wall_n_models(
                "grid_manual_train_scenario_parallel_total",
                scenario_name,
                fallback=get_total_grid_train_models(),
            ),
            "TRAIN Equal": format_or_dash(equal_train_wall),
            "modelos Equal": get_wall_n_models(
                "equal_weights_train_scenario_parallel_total",
                scenario_name,
                fallback=get_total_single_strategy_train_models(),
            ),
            "TRAIN Sklearn": format_or_dash(sklearn_train_wall),
            "modelos Sklearn": get_wall_n_models(
                "sklearn_balanced_train_scenario_parallel_total",
                scenario_name,
                fallback=get_total_single_strategy_train_models(),
            ),
            "Total TRAIN": format_or_dash(total_train_wall),
        })

    print("\n0.1) DESGLOSE DEL ENTRENAMIENTO")
    print("   Weighted grid entrena todos los pares (c0, c1) de la rejilla y guarda coeficientes.")
    print("   Equal weights y Sklearn balanced entrenan un único modelo por seed.")
    print(f"   Rejilla completa: {len(C_VALUES)} x {len(C_VALUES)} = {grid_size} pesos por seed x {n_seeds} seeds.")
    print(pd.DataFrame(train_wall_rows).to_string(index=False))

    # ============================================================
    # 0.2) VALIDACIÓN DE LA REJILLA
    # ============================================================
    grid_val_wall_rows = []

    for scenario_name in scenario_order:
        grid_val_wall = get_wall_seconds("grid_manual_validation_scenario_parallel_total", scenario_name)

        grid_val_wall_rows.append({
            "escenario": scenario_name,
            "VAL rejilla": format_or_dash(grid_val_wall),
            "puntos evaluados": get_wall_n_models("grid_manual_validation_scenario_parallel_total", scenario_name),
        })

    print("\n0.2) VALIDACIÓN DE LA REJILLA")
    print("   No se reentrena: se cargan los coeficientes entrenados en TRAIN y se evalúan todos los pesos en VALIDACIÓN.")
    print("   Este bloque solo aplica a Weighted grid / Best F1.")
    print(pd.DataFrame(grid_val_wall_rows).to_string(index=False))

    # ============================================================
    # 0.3) AGREGACIÓN Y SELECCIÓN BEST F1
    # ============================================================
    aggregation_selection_rows = []

    for scenario_name in scenario_order:
        aggregation_seconds = get_wall_seconds("grid_aggregation_by_scenario", scenario_name)
        selection_seconds = get_wall_seconds("grid_best_selection_by_scenario", scenario_name)
        n_points = get_wall_n_models("grid_best_selection_by_scenario", scenario_name)

        aggregation_selection_rows.append({
            "escenario": scenario_name,
            "agregación mean/std": format_or_dash(aggregation_seconds),
            "selección Best F1": format_or_dash(selection_seconds),
            "total agregación/selección": format_or_dash(safe_sum([aggregation_seconds, selection_seconds])),
            "puntos revisados": n_points if n_points > 0 else grid_size,
            "criterio": "max val_f1_mean",
        })

    print("\n0.3) AGREGACIÓN Y SELECCIÓN BEST F1")
    print(f"   Agrega mean/std entre las {n_seeds} seeds por escenario y selecciona el mayor val_f1_mean.")
    print("   Este tiempo sí se incluye en el coste comparable de Weighted grid / Best F1.")
    print(pd.DataFrame(aggregation_selection_rows).to_string(index=False))

    # ============================================================
    # 0.4) TEST FINAL
    # ============================================================
    test_wall_rows = []

    for scenario_name in scenario_order:
        manual_test_wall = get_wall_seconds("final_test_best_val_f1_grid_scenario_parallel_total", scenario_name)
        equal_test_wall = get_wall_seconds("final_test_equal_weights_scenario_parallel_total", scenario_name)
        sklearn_test_wall = get_wall_seconds("final_test_sklearn_balanced_scenario_parallel_total", scenario_name)
        total_test_wall = safe_sum([manual_test_wall, equal_test_wall, sklearn_test_wall])

        test_wall_rows.append({
            "escenario": scenario_name,
            "TEST Best F1": format_or_dash(manual_test_wall),
            "TEST Equal": format_or_dash(equal_test_wall),
            "TEST Sklearn": format_or_dash(sklearn_test_wall),
            "Total TEST": format_or_dash(total_test_wall),
        })

    print("\n0.4) TEST FINAL")
    print("   En TEST no se selecciona nada y no se reentrena: se cargan coeficientes guardados y se evalúan las estrategias finales.")
    print(pd.DataFrame(test_wall_rows).to_string(index=False))

    # ============================================================
    # 0.5) TOTAL REAL POR ESCENARIO
    # ============================================================
    total_wall_rows = []

    for scenario_name in scenario_order:
        weighted_total = safe_sum([
            get_wall_seconds("grid_manual_train_scenario_parallel_total", scenario_name),
            get_wall_seconds("grid_manual_validation_scenario_parallel_total", scenario_name),
            get_wall_seconds("grid_aggregation_by_scenario", scenario_name),
            get_wall_seconds("grid_best_selection_by_scenario", scenario_name),
            get_wall_seconds("final_test_best_val_f1_grid_scenario_parallel_total", scenario_name),
        ])

        equal_total = safe_sum([
            get_wall_seconds("equal_weights_train_scenario_parallel_total", scenario_name),
            get_wall_seconds("final_test_equal_weights_scenario_parallel_total", scenario_name),
        ])

        sklearn_total = safe_sum([
            get_wall_seconds("sklearn_balanced_train_scenario_parallel_total", scenario_name),
            get_wall_seconds("final_test_sklearn_balanced_scenario_parallel_total", scenario_name),
        ])

        all_block_total = safe_sum([weighted_total, equal_total, sklearn_total])

        total_wall_rows.append({
            "escenario": scenario_name,
            "Weighted Best F1": format_or_dash(weighted_total),
            "Equal weights": format_or_dash(equal_total),
            "Sklearn balanced": format_or_dash(sklearn_total),
            "Bloque completo comparable": format_or_dash(all_block_total),
        })

    print("\n0.5) TOTAL REAL POR ESCENARIO")
    print("   Weighted Best F1 = TRAIN rejilla + VAL rejilla + agregación/selección + TEST Best F1.")
    print("   Equal weights = TRAIN Equal + TEST Equal. Sklearn balanced = TRAIN Sklearn + TEST Sklearn.")
    print("   No incluye figuras ni validación final informativa.")
    print(pd.DataFrame(total_wall_rows).to_string(index=False))

    # ============================================================
    # 0.6) CARGA DE RESULTADOS EXISTENTES, SI APLICA
    # ============================================================
    load_rows = []
    for stage_name, label in [
        ("load_existing_grid_outputs", "Carga resultados existentes de la rejilla"),
        ("load_existing_final_strategy_coefficients", "Carga coeficientes finales existentes"),
        ("load_existing_final_test_outputs", "Carga resultados finales de test existentes"),
    ]:
        seconds = get_wall_seconds(stage_name)
        if not pd.isna(seconds):
            load_rows.append({
                "bloque": label,
                "tiempo real": format_seconds(seconds),
            })

    if load_rows:
        print("\n0.6) CARGA DE RESULTADOS EXISTENTES")
        print("   Estos tiempos solo aparecen cuando el script reutiliza salidas ya guardadas.")
        print(pd.DataFrame(load_rows).to_string(index=False))

    # ============================================================
    # 0.7) OTROS TIEMPOS REALES
    # ============================================================
    other_rows = []
    for stage_name, label in [
        ("figures_total", "Generación de figuras"),
        ("script_total", "Script completo"),
    ]:
        seconds = get_wall_seconds(stage_name)
        if not pd.isna(seconds):
            other_rows.append({
                "bloque": label,
                "tiempo real": format_seconds(seconds),
            })

    if other_rows:
        print("\n0.7) OTROS TIEMPOS REALES")
        print("   Estos tiempos se guardan aparte porque no forman parte del coste experimental comparable hasta TEST.")
        print("   El script completo sí incluye todo lo ejecutado, incluidas figuras, guardado de archivos y validación final informativa.")
        print(pd.DataFrame(other_rows).to_string(index=False))

    print("\nResumen de tiempos comparables guardado en timers_execution.txt.")


def main():
    total_script_start = time.perf_counter()
    timer_frames = []

    print("PROJECT_DIR =", PROJECT_DIR.resolve())
    print("OUTPUT_DIR =", OUTPUT_DIR.resolve())
    print("RAW_RESULTS_PATH =", RAW_RESULTS_PATH.resolve(), RAW_RESULTS_PATH.exists())
    print("RAW_PREDICTIONS_PATH =", RAW_PREDICTIONS_PATH.resolve(), RAW_PREDICTIONS_PATH.exists())
    print("TIMERS_PATH =", TIMERS_PATH.resolve())

    for scenario_name in SCENARIOS.keys():
        for strategy_name in ["grid_manual", "equal_weights", "sklearn_balanced"]:
            p = get_coefficients_wide_path(scenario_name, strategy_name)
            print(p.resolve(), p.exists())

    compatible_outputs = existing_val_grid_outputs_are_compatible()
    print("outputs_compatibles_con_esta_version =", compatible_outputs)

    if RAW_RESULTS_PATH.exists() and not compatible_outputs:
        print("\nSe han detectado resultados antiguos o incompatibles.")
        print("Se eliminan los CSV principales antiguos y se reentrena la rejilla de validación.\n")
        remove_old_incompatible_outputs()

    # ============================================================
    # FASE 1 · TRAIN: ENTRENAR Y GUARDAR COEFICIENTES
    # ============================================================

    balanced_df = None

    if existing_val_grid_outputs_are_compatible():
        t_load_start = time.perf_counter()
        print("\nYa existen resultados, predicciones de validación y coeficientes compatibles.")
        print("Se cargan directamente y se saltan TRAIN/VALIDACIÓN de la rejilla manual.\n")

        results_df = pd.read_csv(RAW_RESULTS_PATH)

        # Aseguramos que los datasets existen para el resto del flujo.
        for scenario_name, config in SCENARIOS.items():
            for seed in SEEDS:
                load_or_create_dataset(seed, scenario_name, config)

        add_wall_clock_timer(
            timer_frames,
            scenario_name="all",
            stage="load_existing_grid_outputs",
            start_time=t_load_start,
            n_models=0,
            skipped=True,
        )

    else:
        run_parallel_warmup()

        print("\n" + "#" * 100)
        print("FASE 1 · TRAIN: entrenamiento y guardado de coeficientes")
        print("#" * 100)

        # 1A) TRAIN de la rejilla manual completa.
        for scenario_name, config in SCENARIOS.items():
            scenario_wall_start = time.perf_counter()

            print("\n" + "=" * 80)
            print(f"TRAIN · REJILLA MANUAL · ESCENARIO: {scenario_name.upper()}")
            print("=" * 80)
            print(f"Ejecutando {len(SEEDS)} semillas en paralelo...")
            print(f"Rejilla: {len(C_VALUES)} x {len(C_VALUES)} = {len(C_VALUES) * len(C_VALUES)} modelos por semilla")

            parallel_output = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_seed_grid_manual_train)(seed, scenario_name, config)
                for seed in SEEDS
            )

            scenario_coeffs = [item[0] for item in parallel_output]
            scenario_timers = [item[1] for item in parallel_output]

            scenario_coeffs_df = pd.concat(scenario_coeffs, ignore_index=True)
            scenario_timers_df = pd.concat(scenario_timers, ignore_index=True)
            timer_frames.append(scenario_timers_df)

            n_grid_train_models = int(scenario_timers_df["n_models"].fillna(0).sum())
            n_grid_train_warnings = int(scenario_timers_df["convergence_warnings"].fillna(0).sum()) if "convergence_warnings" in scenario_timers_df.columns else 0
            n_grid_train_max_iter = int(scenario_timers_df["reached_max_iter_count"].fillna(0).sum()) if "reached_max_iter_count" in scenario_timers_df.columns else 0
            print(
                f"Convergencia TRAIN rejilla {scenario_name}: "
                f"warnings={n_grid_train_warnings}/{n_grid_train_models}, "
                f"alcanzan max_iter={n_grid_train_max_iter}/{n_grid_train_models}"
            )

            scenario_coeffs_df = scenario_coeffs_df.sort_values(
                ["scenario", "seed", "c0", "c1"]
            ).reset_index(drop=True)

            scenario_coeffs_df.to_csv(
                get_coefficients_wide_path(scenario_name, "grid_manual"),
                index=False,
            )

            add_wall_clock_timer(
                timer_frames,
                scenario_name=scenario_name,
                stage="grid_manual_train_scenario_parallel_total",
                start_time=scenario_wall_start,
                n_models=len(SEEDS) * len(C_VALUES) * len(C_VALUES),
                skipped=False,
            )

            print(f"TRAIN de rejilla manual completado para escenario {scenario_name}.")

        # 1B) TRAIN de estrategias finales independientes.
        balanced_df = compute_balanced_points_by_scenario()
        train_or_load_final_strategy_coefficients(timer_frames)

        # ============================================================
        # FASE 2 · VALIDACIÓN: CARGAR COEFICIENTES Y EVALUAR
        # ============================================================
        print("\n" + "#" * 100)
        print("FASE 2 · VALIDACIÓN: evaluación de la rejilla manual con coeficientes guardados")
        print("#" * 100)

        for scenario_name, config in SCENARIOS.items():
            scenario_wall_start = time.perf_counter()

            print("\n" + "=" * 80)
            print(f"VALIDACIÓN · REJILLA MANUAL · ESCENARIO: {scenario_name.upper()}")
            print("=" * 80)
            print("No se entrena: se cargan coeficientes entrenados en TRAIN y se evalúan en VALIDACIÓN.")

            parallel_output = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_seed_grid_manual_validation)(seed, scenario_name, config)
                for seed in SEEDS
            )

            scenario_results = [item[0] for item in parallel_output]
            scenario_predictions = [item[1] for item in parallel_output]
            scenario_timers = [item[2] for item in parallel_output]

            scenario_results_df = pd.concat(scenario_results, ignore_index=True)
            scenario_predictions_df = pd.concat(scenario_predictions, ignore_index=True)
            scenario_timers_df = pd.concat(scenario_timers, ignore_index=True)

            all_results_list.append(scenario_results_df)
            all_predictions_list.append(scenario_predictions_df)
            timer_frames.append(scenario_timers_df)

            add_wall_clock_timer(
                timer_frames,
                scenario_name=scenario_name,
                stage="grid_manual_validation_scenario_parallel_total",
                start_time=scenario_wall_start,
                n_models=len(SEEDS) * len(C_VALUES) * len(C_VALUES),
                skipped=False,
            )

            print(f"VALIDACIÓN de rejilla manual completada para escenario {scenario_name}.")

        results_df = pd.concat(all_results_list, ignore_index=True)
        predictions_df = pd.concat(all_predictions_list, ignore_index=True)

        results_df = results_df.sort_values(
            ["scenario", "seed", "c0", "c1"]
        ).reset_index(drop=True)

        predictions_df = predictions_df.sort_values(
            ["scenario", "seed", "sample_id", "c0", "c1"]
        ).reset_index(drop=True)

        results_df.to_csv(RAW_RESULTS_PATH, index=False)
        predictions_df.to_csv(RAW_PREDICTIONS_PATH, index=False)

        print("\nResultados raw de la rejilla guardados en:")
        print(RAW_RESULTS_PATH)
        print(RAW_PREDICTIONS_PATH)

    duplicated_like_cols = [
        col for col in results_df.columns
        if col.endswith("_x") or col.endswith("_y")
    ]

    results_df = results_df.drop(columns=duplicated_like_cols, errors="ignore")

    results_df = results_df.sort_values(
        ["scenario", "seed", "c0", "c1"]
    ).reset_index(drop=True)

    results_df.to_csv(RAW_RESULTS_PATH, index=False)

    # Agregado de la rejilla de VALIDACIÓN.
    # Se mide por escenario para que el coste comparable de Weighted grid incluya
    # también el postprocesado necesario para seleccionar Best F1.
    group_cols = ["scenario", "c0", "c1"]
    metric_cols = [
        col for col in results_df.columns
        if col not in ["scenario", "seed", "c0", "c1", "convergence_message"]
        and pd.api.types.is_numeric_dtype(results_df[col])
    ]

    agg_frames = []

    for scenario_name in SCENARIOS.keys():
        t_grid_aggregation_scenario_start = time.perf_counter()

        scenario_results_df = results_df[
            results_df["scenario"] == scenario_name
        ].copy()

        scenario_agg_df = (
            scenario_results_df
            .groupby(group_cols)[metric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )

        new_columns = []

        for col in scenario_agg_df.columns:
            if col[1] == "":
                new_columns.append(col[0])
            else:
                new_columns.append(f"{col[0]}_{col[1]}")

        scenario_agg_df.columns = new_columns

        std_cols = [col for col in scenario_agg_df.columns if col.endswith("_std")]
        scenario_agg_df[std_cols] = scenario_agg_df[std_cols].fillna(0.0)

        scenario_agg_df = scenario_agg_df.sort_values(
            ["scenario", "c0", "c1"]
        ).reset_index(drop=True)

        agg_frames.append(scenario_agg_df)

        add_wall_clock_timer(
            timer_frames,
            scenario_name=scenario_name,
            stage="grid_aggregation_by_scenario",
            start_time=t_grid_aggregation_scenario_start,
            n_models=len(scenario_results_df),
            skipped=False,
        )

    agg_df = pd.concat(agg_frames, ignore_index=True)

    agg_df = agg_df.sort_values(
        ["scenario", "c0", "c1"]
    ).reset_index(drop=True)

    agg_df.to_csv(AGG_RESULTS_PATH, index=False)

    print("\nResultados agregados de la rejilla de validación guardados en:")
    print(AGG_RESULTS_PATH)

    # Mejor configuración por escenario según val_f1_mean.
    # La selección se desglosa por escenario para que la terminal muestre tiempos reales por escenario.
    best_rows = []

    for scenario_name in SCENARIOS.keys():
        t_selection_scenario_start = time.perf_counter()
        scenario_df = agg_df[agg_df["scenario"] == scenario_name].copy()
        idx_best = scenario_df["val_f1_mean"].idxmax()
        best_row = scenario_df.loc[idx_best]
        best_rows.append(best_row)

        add_wall_clock_timer(
            timer_frames,
            scenario_name=scenario_name,
            stage="grid_best_selection_by_scenario",
            start_time=t_selection_scenario_start,
            n_models=len(scenario_df),
            skipped=False,
        )

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(BEST_CONFIGS_PATH, index=False)


    if balanced_df is None:
        balanced_df = compute_balanced_points_by_scenario()
        train_or_load_final_strategy_coefficients(timer_frames)

    cols_best = [
        "scenario",
        "c0",
        "c1",
        "val_f1_mean",
        "val_f1_std",
        "val_pr_auc_mean",
        "val_pr_auc_std",
        "val_balanced_accuracy_mean",
        "val_balanced_accuracy_std",
        "val_mcc_mean",
        "val_mcc_std",
        "val_recall_mean",
        "val_recall_std",
        "val_fnr_mean",
        "val_fnr_std",
        "val_specificity_mean",
        "val_specificity_std",
        "val_roc_auc_mean",
        "val_roc_auc_std",
    ]

    cols_best_existing = [col for col in cols_best if col in best_df.columns]

    print("\nMejor configuración según val_f1_mean en VALIDACIÓN:")
    best_to_print = best_df[cols_best_existing].copy()

    for idx, row in best_to_print.iterrows():
        print("\n" + "=" * 60)
        print(f"Fila en la tabla agregada de validación: {idx}")
        print("=" * 60)
        print(row.to_string())

    print("\nPuntos equivalentes a class_weight='balanced' de Scikit-Learn:")
    print(balanced_df.to_string(index=False))

    # ============================================================
    # TABLA COMPARATIVA FINAL EN VALIDACIÓN
    # ============================================================

    if raw_final_outputs_have_expected_scenarios(VAL_FINAL_RAW_RESULTS_PATH) and VAL_FINAL_RAW_PREDICTIONS_PATH.exists():
        t_val_load_start = time.perf_counter()
        print("\nYa existen resultados finales de validación.")
        print("Se cargan directamente y se salta la evaluación final en validación.\n")
        val_final_results_df = pd.read_csv(VAL_FINAL_RAW_RESULTS_PATH)
        add_wall_clock_timer(
            timer_frames,
            scenario_name="all",
            stage="load_existing_final_validation_outputs",
            start_time=t_val_load_start,
            n_models=0,
            skipped=True,
        )
    else:
        if VAL_FINAL_RAW_RESULTS_PATH.exists():
            VAL_FINAL_RAW_RESULTS_PATH.unlink()
        if VAL_FINAL_RAW_PREDICTIONS_PATH.exists():
            VAL_FINAL_RAW_PREDICTIONS_PATH.unlink()

        all_val_results = []
        all_val_predictions = []

        print("\n" + "#" * 100)
        print("FASE 2 · VALIDACIÓN FINAL: evaluación separada por estrategia")
        print("#" * 100)

        for scenario_name, config in SCENARIOS.items():
            best_row = best_df[best_df["scenario"] == scenario_name].iloc[0]

            best_c0 = float(best_row["c0"])
            best_c1 = float(best_row["c1"])

            for model_name in ["best_val_f1_grid", "equal_weights", "sklearn_balanced"]:
                val_wall_start = time.perf_counter()

                print("\n" + "=" * 80)
                print(f"VALIDACIÓN FINAL · {model_name.upper()} · ESCENARIO: {scenario_name.upper()}")

                if model_name == "best_val_f1_grid":
                    print(
                        f"Pesos manuales seleccionados según val_f1_mean: "
                        f"c0={best_c0}, c1={best_c1}"
                    )
                    print("Esta estrategia usa los pesos manuales seleccionados en validación.")

                elif model_name == "equal_weights":
                    print(
                        f"Pesos de la estrategia Equal weights: "
                        f"c0={EQUAL_WEIGHTS_C0}, c1={EQUAL_WEIGHTS_C1}"
                    )

                elif model_name == "sklearn_balanced":
                    balanced_row = balanced_df[balanced_df["scenario"] == scenario_name].iloc[0]
                    print("Pesos de la estrategia Sklearn balanced: calculados automáticamente por seed a partir de y_train.")
                    print(
                        f"Pesos usados: "
                        f"c0={balanced_row['balanced_c0_mean']:.6g}, "
                        f"c1={balanced_row['balanced_c1_mean']:.6g} "
                        f"(std entre seeds: c0={balanced_row['balanced_c0_std']:.6g}, "
                        f"c1={balanced_row['balanced_c1_std']:.6g})"
                    )

                print("No se entrena: se cargan coeficientes guardados en TRAIN y se evalúa en VALIDACIÓN.")
                print("=" * 80)

                parallel_output = Parallel(n_jobs=N_JOBS)(
                    delayed(process_one_seed_final_single_model)(
                        seed,
                        scenario_name,
                        config,
                        best_c0,
                        best_c1,
                        "val",
                        model_name,
                    )
                    for seed in SEEDS
                )

                scenario_val_results = [item[0] for item in parallel_output]
                scenario_val_predictions = [item[1] for item in parallel_output]
                scenario_val_timers = [item[2] for item in parallel_output]

                scenario_val_results_df = pd.concat(scenario_val_results, ignore_index=True)
                scenario_val_predictions_df = pd.concat(scenario_val_predictions, ignore_index=True)
                scenario_val_timers_df = pd.concat(scenario_val_timers, ignore_index=True)

                all_val_results.append(scenario_val_results_df)
                all_val_predictions.append(scenario_val_predictions_df)
                timer_frames.append(scenario_val_timers_df)

                add_wall_clock_timer(
                    timer_frames,
                    scenario_name=scenario_name,
                    stage=f"final_val_{model_name}_scenario_parallel_total",
                    start_time=val_wall_start,
                    n_models=len(SEEDS),
                    skipped=False,
                )

                print(f"VALIDACIÓN FINAL de {model_name} completada para escenario {scenario_name}.")

        val_final_results_df = pd.concat(all_val_results, ignore_index=True)
        val_final_predictions_df = pd.concat(all_val_predictions, ignore_index=True)

        val_final_results_df = val_final_results_df.sort_values(
            ["scenario", "model_name", "seed"]
        ).reset_index(drop=True)

        val_final_predictions_df = val_final_predictions_df.sort_values(
            ["scenario", "model_name", "seed", "sample_id"]
        ).reset_index(drop=True)

        val_final_results_df.to_csv(VAL_FINAL_RAW_RESULTS_PATH, index=False)
        val_final_predictions_df.to_csv(VAL_FINAL_RAW_PREDICTIONS_PATH, index=False)

        print("\nResultados finales de validación guardados en:")
        print(VAL_FINAL_RAW_RESULTS_PATH)
        print(VAL_FINAL_RAW_PREDICTIONS_PATH)

    val_agg_df = aggregate_final_results(
        val_final_results_df,
        group_cols=["scenario", "model_name"],
    )

    val_agg_df.to_csv(VAL_FINAL_AGG_RESULTS_PATH, index=False)

    val_comparison_df = make_comparison_table(val_agg_df, split_name="val")
    val_comparison_df.to_csv(VAL_FINAL_COMPARISON_PATH, index=False)

    print_comparison_table_by_scenario(
        val_comparison_df,
        title="Tabla comparativa final en VALIDACIÓN",
        split_name="val",
        decimals=4,
    )

    print("\nResultados finales de validación agregados guardados en:")
    print(VAL_FINAL_AGG_RESULTS_PATH)
    print(VAL_FINAL_COMPARISON_PATH)

    # ============================================================
    # TABLA COMPARATIVA FINAL EN TEST
    # ============================================================

    if raw_final_outputs_have_expected_scenarios(TEST_FINAL_RAW_RESULTS_PATH) and TEST_FINAL_RAW_PREDICTIONS_PATH.exists():
        t_test_load_start = time.perf_counter()
        print("\nYa existen resultados finales de test.")
        print("Se cargan directamente y se salta la evaluación final en test.\n")
        test_final_results_df = pd.read_csv(TEST_FINAL_RAW_RESULTS_PATH)
        add_wall_clock_timer(
            timer_frames,
            scenario_name="all",
            stage="load_existing_final_test_outputs",
            start_time=t_test_load_start,
            n_models=0,
            skipped=True,
        )
    else:
        if TEST_FINAL_RAW_RESULTS_PATH.exists():
            TEST_FINAL_RAW_RESULTS_PATH.unlink()
        if TEST_FINAL_RAW_PREDICTIONS_PATH.exists():
            TEST_FINAL_RAW_PREDICTIONS_PATH.unlink()

        all_test_results = []
        all_test_predictions = []

        print("\n" + "#" * 100)
        print("FASE 3 · TEST FINAL: evaluación separada por estrategia")
        print("#" * 100)

        for scenario_name, config in SCENARIOS.items():
            best_row = best_df[best_df["scenario"] == scenario_name].iloc[0]

            best_c0 = float(best_row["c0"])
            best_c1 = float(best_row["c1"])

            for model_name in ["best_val_f1_grid", "equal_weights", "sklearn_balanced"]:
                test_wall_start = time.perf_counter()

                print("\n" + "=" * 80)
                print(f"TEST FINAL · {model_name.upper()} · ESCENARIO: {scenario_name.upper()}")

                if model_name == "best_val_f1_grid":
                    print(
                        f"Pesos manuales seleccionados previamente en validación: "
                        f"c0={best_c0}, c1={best_c1}"
                    )
                    print("Esta estrategia usa esos pesos manuales para la evaluación final en TEST.")

                elif model_name == "equal_weights":
                    print(
                        f"Pesos de la estrategia Equal weights: "
                        f"c0={EQUAL_WEIGHTS_C0}, c1={EQUAL_WEIGHTS_C1}"
                    )

                elif model_name == "sklearn_balanced":
                    balanced_row = balanced_df[balanced_df["scenario"] == scenario_name].iloc[0]
                    print("Pesos de la estrategia Sklearn balanced: calculados automáticamente por seed a partir de y_train.")
                    print(
                        f"Pesos usados: "
                        f"c0={balanced_row['balanced_c0_mean']:.6g}, "
                        f"c1={balanced_row['balanced_c1_mean']:.6g} "
                        f"(std entre seeds: c0={balanced_row['balanced_c0_std']:.6g}, "
                        f"c1={balanced_row['balanced_c1_std']:.6g})"
                    )

                print("No se entrena ni se selecciona nada: se cargan coeficientes guardados y se evalúa en TEST.")
                print("=" * 80)

                parallel_output = Parallel(n_jobs=N_JOBS)(
                    delayed(process_one_seed_final_single_model)(
                        seed,
                        scenario_name,
                        config,
                        best_c0,
                        best_c1,
                        "test",
                        model_name,
                    )
                    for seed in SEEDS
                )

                scenario_test_results = [item[0] for item in parallel_output]
                scenario_test_predictions = [item[1] for item in parallel_output]
                scenario_test_timers = [item[2] for item in parallel_output]

                scenario_test_results_df = pd.concat(scenario_test_results, ignore_index=True)
                scenario_test_predictions_df = pd.concat(scenario_test_predictions, ignore_index=True)
                scenario_test_timers_df = pd.concat(scenario_test_timers, ignore_index=True)

                all_test_results.append(scenario_test_results_df)
                all_test_predictions.append(scenario_test_predictions_df)
                timer_frames.append(scenario_test_timers_df)

                add_wall_clock_timer(
                    timer_frames,
                    scenario_name=scenario_name,
                    stage=f"final_test_{model_name}_scenario_parallel_total",
                    start_time=test_wall_start,
                    n_models=len(SEEDS),
                    skipped=False,
                )

                print(f"TEST FINAL de {model_name} completado para escenario {scenario_name}.")

        test_final_results_df = pd.concat(all_test_results, ignore_index=True)
        test_final_predictions_df = pd.concat(all_test_predictions, ignore_index=True)

        test_final_results_df = test_final_results_df.sort_values(
            ["scenario", "model_name", "seed"]
        ).reset_index(drop=True)

        test_final_predictions_df = test_final_predictions_df.sort_values(
            ["scenario", "model_name", "seed", "sample_id"]
        ).reset_index(drop=True)

        test_final_results_df.to_csv(TEST_FINAL_RAW_RESULTS_PATH, index=False)
        test_final_predictions_df.to_csv(TEST_FINAL_RAW_PREDICTIONS_PATH, index=False)

        print("\nResultados finales de test guardados en:")
        print(TEST_FINAL_RAW_RESULTS_PATH)
        print(TEST_FINAL_RAW_PREDICTIONS_PATH)

    test_agg_df = aggregate_final_results(
        test_final_results_df,
        group_cols=["scenario", "model_name"],
    )

    test_agg_df.to_csv(TEST_FINAL_AGG_RESULTS_PATH, index=False)

    test_comparison_df = make_comparison_table(test_agg_df, split_name="test")
    test_comparison_df.to_csv(TEST_FINAL_COMPARISON_PATH, index=False)

    print_comparison_table_by_scenario(
        test_comparison_df,
        title="Tabla comparativa final en TEST",
        split_name="test",
        decimals=4,
    )

    print("\nResultados finales de test agregados guardados en:")
    print(TEST_FINAL_AGG_RESULTS_PATH)
    print(TEST_FINAL_COMPARISON_PATH)

    # Figuras.
    t_fig_start = time.perf_counter()
    save_validation_figures(agg_df, best_df, balanced_df, val_comparison_df)
    save_val_final_comparison_plots(val_agg_df)
    save_test_final_comparison_plots(test_agg_df)
    save_final_confusion_matrix_panels(VAL_FINAL_RAW_PREDICTIONS_PATH, split_name="val")
    save_final_confusion_matrix_panels(TEST_FINAL_RAW_PREDICTIONS_PATH, split_name="test")
    save_final_boxplot_plots_and_tables(val_final_results_df, split_name="val")
    save_final_boxplot_plots_and_tables(test_final_results_df, split_name="test")
    add_wall_clock_timer(
        timer_frames,
        scenario_name="all",
        stage="figures_total",
        start_time=t_fig_start,
        n_models=0,
        skipped=False,
    )

    add_wall_clock_timer(
        timer_frames,
        scenario_name="all",
        stage="script_total",
        start_time=total_script_start,
        n_models=np.nan,
        skipped=False,
    )

    write_experiment_metadata()
    timers_df = save_timers(timer_frames)

    import io
    from contextlib import redirect_stdout

    timer_summary_buffer = io.StringIO()
    with redirect_stdout(timer_summary_buffer):
        print_clean_timer_summary(timers_df)

    timer_summary_text = timer_summary_buffer.getvalue()
    print(timer_summary_text, end="")

    with open(TIMERS_PATH, "w", encoding="utf-8") as f:
        f.write(timer_summary_text)

    print("\nResumen de tiempos guardado en:")
    print(TIMERS_PATH)

    print("\nTodo terminado correctamente.")
    print("Resultados, datasets, predicciones de validación, coeficientes, test, timers y figuras guardados en:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
