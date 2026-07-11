
import json
import time
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from joblib import Parallel, delayed
from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, make_scorer)
from imblearn.metrics import (sensitivity_score, specificity_score, geometric_mean_score, make_index_balanced_accuracy)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from greedy_logreg import GreedyClassWeightLogisticRegressionCV

# ============================================================
# 0. CONFIGURACIÓN GENERAL
# ============================================================

SEEDS = list(range(1, 21))
N_JOBS = -1
MODEL_MAX_ITER = 500
COMPARISON_RUN_SIGNATURE = "comparison_resampling_train_real_v3_warmup_completo"

# Estrategias que ya vienen de los códigos base WBCE.
BASE_MODEL_NAMES = ["best_val_f1_grid", "equal_weights", "sklearn_balanced"]

# Nueva estrategia heurística de búsqueda de pesos de clase.
# La búsqueda se realiza SOLO dentro del TRAIN mediante CV interna estratificada de 5 folds.
GREEDY_MODEL_NAMES = [
    "greedy_class_weight",
]

# Nuevas estrategias clásicas de tratamiento del desbalanceo.
# Todas se aplican SOLO al TRAIN. Después se entrena una LogisticRegression sin class_weight.
RESAMPLING_MODEL_NAMES = [
    "random_oversampling",
    "smote",
    "random_undersampling",
]

# Estrategias entrenadas en este script de comparación.
TRAINED_COMPARISON_MODEL_NAMES = GREEDY_MODEL_NAMES + RESAMPLING_MODEL_NAMES

# Estrategias usadas únicamente para el warm-up técnico.
# Se mantienen separadas de RESAMPLING_MODEL_NAMES para que, si en el futuro
# se añade una nueva estrategia con parámetros especiales, el warm-up no falle
# por intentar ejecutarla automáticamente.
WARMUP_RESAMPLING_MODEL_NAMES = [
    "random_oversampling",
    "smote",
    "random_undersampling",
]

ALL_MODEL_NAMES = ["best_val_f1_grid", "greedy_class_weight", "equal_weights", "sklearn_balanced"] + RESAMPLING_MODEL_NAMES

PREFERRED_PLOT_ORDER = [
    "equal_weights",
    "best_val_f1_grid",
    "greedy_class_weight",
    "sklearn_balanced",
    "random_oversampling",
    "random_undersampling",
    "smote",
]

MODEL_COLORS = {
    "best_val_f1_grid": "#D55E00",
    "greedy_class_weight": "#332288",
    "equal_weights": "#CC79A7",
    "sklearn_balanced": "#0072B2",
    "random_oversampling": "#009E73",
    "smote": "#56B4E9",
    "random_undersampling": "#999999",
}

MODEL_LABELS = {
    "best_val_f1_grid": "WBCE Best F1",
    "greedy_class_weight": "WBCE Heurística",
    "equal_weights": "Equal weights",
    "sklearn_balanced": "Sklearn balanced",
    "random_oversampling": "Random Oversampling",
    "smote": "SMOTE",
    "random_undersampling": "Random Undersampling",
}

PROJECT_DIR = Path(__file__).resolve().parent


def stop_missing_base_flow(folder_name, required_files, searched_roots=None, extra_message=""):
    """
    Detiene el script con un mensaje claro cuando no están los outputs del flujo base.
    """
    print("\n" + "=" * 110)
    print("NO SE HAN ENCONTRADO LOS RESULTADOS BASE NECESARIOS")
    print("=" * 110)
    print(f"Carpeta de outputs esperada: {folder_name}")
    print("Archivos mínimos necesarios:")
    for file_name in required_files:
        print(f"  - {file_name}")
    print("\nQué hacer:")
    print("  1) Ejecuta primero el código WBCE/base correspondiente hasta que termine correctamente.")
    print("  2) Comprueba que se haya generado la carpeta de outputs anterior con esos CSV.")
    print("  3) Después vuelve a ejecutar este código de comparación de estrategias.")
    if extra_message:
        print("\nDetalle:")
        print(extra_message)
    if searched_roots:
        print("\nRutas revisadas automáticamente:")
        for root in searched_roots:
            print(f"  - {root}")
    print("=" * 110 + "\n")
    raise SystemExit(1)


def find_base_output_dir(folder_name, required_files):
    """
    Busca automáticamente la carpeta de outputs del código base.

    No depende del nombre de la carpeta principal del proyecto. Primero busca en
    la carpeta donde está este script y luego va subiendo por carpetas padre.
    En cada nivel mira también dentro de carpetas hermanas, que es la organización
    habitual del proyecto:

    carpeta_proyecto/
        Estrategia_WBCE_Datasets_Sinteticos/outputs_...
        Estrategia_WBCE_Dataset_Real_1/outputs_...
        Compararciones_Globales_con_Estrategias_Clasicas_de_Desbalanceo/...
    """
    search_roots = [PROJECT_DIR] + list(PROJECT_DIR.parents)
    checked_roots = []

    for root in search_roots:
        checked_roots.append(root)

        # 1) Carpeta de outputs directamente en el nivel actual.
        candidate = root / folder_name
        if candidate.exists() and candidate.is_dir():
            if all((candidate / file_name).exists() for file_name in required_files):
                return candidate

        # 2) Carpeta de outputs dentro de una carpeta hermana.
        try:
            for subdir in root.iterdir():
                if not subdir.is_dir():
                    continue

                candidate = subdir / folder_name
                if candidate.exists() and candidate.is_dir():
                    if all((candidate / file_name).exists() for file_name in required_files):
                        return candidate
        except (PermissionError, OSError):
            continue

    stop_missing_base_flow(
        folder_name=folder_name,
        required_files=required_files,
        searched_roots=checked_roots,
        extra_message=(
            "No se ha encontrado una carpeta de outputs compatible. "
            "No hace falta que la carpeta principal del proyecto tenga un nombre concreto, "
            "pero los outputs generados por el código base sí deben estar dentro del árbol del proyecto."
        ),
    )

iba_metric = make_index_balanced_accuracy(alpha=0.1, squared=True)(geometric_mean_score)

def _unique_weight_evaluations(cv_results):
    """Cuenta pares distintos (c0, c1) evaluados por GreedyClassWeightLogisticRegressionCV."""
    points = set()
    try:
        for round_dict in cv_results:
            for c0_value, c1_value in round_dict.get("weights", []):
                points.add((round(float(c0_value), 8), round(float(c1_value), 8)))
    except Exception:
        return np.nan
    return len(points)


def _get_greedy_final_estimator(greedy_model):
    """Devuelve el estimador final reajustado si la librería lo expone con otro nombre."""
    for attr_name in ["estimator_", "model_", "best_estimator_", "clf_", "logistic_regression_"]:
        if hasattr(greedy_model, attr_name):
            estimator = getattr(greedy_model, attr_name)
            if estimator is not None and hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
                return estimator
    return greedy_model


def _extract_class_weight_value(class_weight_dict, class_label):
    """Extrae un peso de clase admitiendo claves enteras o texto."""
    if class_weight_dict is None:
        return np.nan
    if class_label in class_weight_dict:
        return float(class_weight_dict[class_label])
    str_label = str(class_label)
    if str_label in class_weight_dict:
        return float(class_weight_dict[str_label])
    return np.nan


def get_base_grid_size_for_scenario(scenario_name):
    """
    Recupera automáticamente el tamaño de la rejilla del código base WBCE.
    """
    scenarios_config_path = BASE_OUTPUT_DIR / "scenarios_config.csv"

    if scenarios_config_path.exists():
        try:
            config_df = pd.read_csv(scenarios_config_path)

            if "scenario" in config_df.columns:
                config_df = config_df[config_df["scenario"].astype(str) == str(scenario_name)]

            if not config_df.empty and "grid_size" in config_df.columns:
                value = pd.to_numeric(config_df["grid_size"], errors="coerce").dropna()
                if not value.empty:
                    return int(value.iloc[0])

            if not config_df.empty and "n_c_values" in config_df.columns:
                value = pd.to_numeric(config_df["n_c_values"], errors="coerce").dropna()
                if not value.empty:
                    return int(value.iloc[0]) * int(value.iloc[0])

        except Exception:
            pass

    return np.nan


def get_base_train_models_for_strategy(model_name, scenario_name):
    """
    Número total de modelos entrenados en TRAIN por las estrategias del código base WBCE.
    """
    if model_name == "best_val_f1_grid":
        grid_size = get_base_grid_size_for_scenario(scenario_name)
        if not pd.isna(grid_size):
            return int(len(SEEDS) * int(grid_size))
        return np.nan

    if model_name in ["equal_weights", "sklearn_balanced"]:
        return int(len(SEEDS))

    return np.nan


def get_greedy_train_models_from_coefficients(scenario_name):
    """
    Calcula el número exacto de modelos entrenados por la heurística greedy
    usando el CSV de coeficientes guardado.
    """
    try:
        path = get_resampling_coefficients_path(scenario_name, "greedy_class_weight")
        if not path.exists():
            return np.nan

        coeff_df = pd.read_csv(path)
        coeff_df = coeff_df[
            (coeff_df["scenario"].astype(str) == str(scenario_name))
            & (coeff_df["model_name"].astype(str) == "greedy_class_weight")
        ].copy()

        if coeff_df.empty or "greedy_weight_pairs_evaluated" not in coeff_df.columns:
            return np.nan

        pairs = pd.to_numeric(coeff_df["greedy_weight_pairs_evaluated"], errors="coerce")
        if "greedy_cv_folds" in coeff_df.columns:
            folds = pd.to_numeric(coeff_df["greedy_cv_folds"], errors="coerce").fillna(5)
        else:
            folds = pd.Series([5] * len(coeff_df), index=coeff_df.index)

        valid = pairs.notna()
        if valid.sum() == 0:
            return np.nan

        return int((pairs[valid] * folds[valid] + 1).sum())

    except Exception:
        return np.nan


def get_expected_train_models_for_strategy(strategy_name, scenario_name):
    """
    Fallback automático para mostrar modelos entrenados en TRAIN aunque se reutilicen salidas existentes.
    """
    if strategy_name in BASE_MODEL_NAMES:
        return get_base_train_models_for_strategy(strategy_name, scenario_name)

    if strategy_name == "greedy_class_weight":
        greedy_models = get_greedy_train_models_from_coefficients(scenario_name)
        if not pd.isna(greedy_models):
            return int(greedy_models)
        return int(len(SEEDS))

    if strategy_name in RESAMPLING_MODEL_NAMES:
        return int(len(SEEDS))

    return np.nan

# ============================================================
# 1. CONFIGURACIÓN REAL 1 · CUANTITATIVAS + CATEGÓRICAS
# ============================================================

DATA_FILE_NAME = "DF1_HBU_JuanSoler.xlsx"
EXCEL_SHEET_NAME = 0
TARGET_COL = "NOS5"
REAL_SCENARIO_NAME = "real_1_cuantitativas_categoricas"
MAX_CATEGORICAL_UNIQUE_VALUES = 30
REFERENCE_CATEGORIES = {"LCA_REUS": "High Satisfaction"}
DATA_FILE_PATH = PROJECT_DIR / DATA_FILE_NAME

BASE_OUTPUT_DIR = find_base_output_dir(
    folder_name="outputs_dataset_real_1_cuantitativas_categoricas",
    required_files=[
        'val_final_results_raw_multiseed.csv',
        'val_final_predictions_selected_models_raw_multiseed.csv',
        'test_final_results_raw_multiseed.csv',
        'test_final_predictions_raw_multiseed.csv',
        'timers_execution.txt',
    ],
)
BASE_DATASETS_DIR = BASE_OUTPUT_DIR / "datasets"
BASE_VAL_RESULTS_PATH = BASE_OUTPUT_DIR / "val_final_results_raw_multiseed.csv"
BASE_VAL_PREDICTIONS_PATH = BASE_OUTPUT_DIR / "val_final_predictions_selected_models_raw_multiseed.csv"
BASE_TEST_RESULTS_PATH = BASE_OUTPUT_DIR / "test_final_results_raw_multiseed.csv"
BASE_TEST_PREDICTIONS_PATH = BASE_OUTPUT_DIR / "test_final_predictions_raw_multiseed.csv"

OUTPUT_DIR = PROJECT_DIR / "comparacion_estrategias_real1_cuanti_categ"
FIGURES_DIR = OUTPUT_DIR / "figures" / "comparativas_globales"
COEFFICIENTS_DIR = OUTPUT_DIR / "coeficientes"
RESAMPLING_INFO_DIR = OUTPUT_DIR / "resampling_info"
TIMES_DIR = OUTPUT_DIR / "tiempos"

SCENARIOS = {
    REAL_SCENARIO_NAME: {
        "dataset": DATA_FILE_NAME,
        "target": TARGET_COL,
        "type": "real_dataset_quantitative_categorical_binary_and_one_hot_reference",
    }
}

def validate_required_base_flow():
    """
    Comprueba que existen los datasets y CSV necesarios del flujo WBCE Real 1 cuantitativas+categóricas.
    Si falta algo, detiene el script antes de entrenar las estrategias nuevas.
    """
    missing = []
    short_name = get_short_scenario_name(REAL_SCENARIO_NAME)

    for path in [
        BASE_VAL_RESULTS_PATH,
        BASE_VAL_PREDICTIONS_PATH,
        BASE_TEST_RESULTS_PATH,
        BASE_TEST_PREDICTIONS_PATH,
        BASE_DATASETS_DIR / "real1_cuant_categ_clean.csv",
    ]:
        if not path.exists():
            missing.append(path)

    for seed in SEEDS:
        dataset_path = BASE_DATASETS_DIR / short_name / f"real1_cuant_categ_seed{seed:02d}.csv"
        if not dataset_path.exists():
            missing.append(dataset_path)

    if missing:
        stop_missing_base_flow(
            folder_name="outputs_dataset_real_1_cuantitativas_categoricas",
            required_files=[
                "val_final_results_raw_multiseed.csv",
                "val_final_predictions_selected_models_raw_multiseed.csv",
                "test_final_results_raw_multiseed.csv",
                "test_final_predictions_raw_multiseed.csv",
                "datasets/real1_cuant_categ_clean.csv",
                "datasets/real1_cuant_categ/real1_cuant_categ_seedXX.csv",
            ],
            searched_roots=[PROJECT_DIR] + list(PROJECT_DIR.parents),
            extra_message=(
                "Se ha localizado o intentado localizar el flujo base, pero faltan archivos concretos.\n"
                "Primeros archivos ausentes:\n  - " + "\n  - ".join(str(p) for p in missing[:12])
            ),
        )

    print("Carpeta base WBCE encontrada:")
    print(BASE_OUTPUT_DIR.resolve())


_CLEAN_DF_CACHE = None
_FEATURE_COLS_CACHE = None


def get_scenarios():
    return SCENARIOS


def get_target_col():
    return TARGET_COL


def get_short_scenario_name(scenario_name):
    return "real1_cuant_categ" if scenario_name == REAL_SCENARIO_NAME else str(scenario_name).replace(" ", "_")[:24]


def _clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    duplicated_cols = pd.Series(df.columns)[pd.Series(df.columns).duplicated()].tolist()
    if duplicated_cols:
        raise ValueError(f"Hay columnas duplicadas tras limpiar nombres: {duplicated_cols}")
    return df


def _coerce_binary_target(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.Series(s.dropna().unique()).tolist())
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            return s.astype(int)
    mapping = {"0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1, "negative": 0, "positive": 1, "negativo": 0, "positivo": 1}
    mapped = s.astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any():
        raise ValueError(f"No se puede convertir {TARGET_COL} a 0/1. Valores: {sorted(s.dropna().unique().tolist())}")
    return mapped.astype(int)


def load_and_clean_real_dataset():
    global _CLEAN_DF_CACHE, _FEATURE_COLS_CACHE
    if _CLEAN_DF_CACHE is not None and _FEATURE_COLS_CACHE is not None:
        return _CLEAN_DF_CACHE.copy(), list(_FEATURE_COLS_CACHE)

    clean_path = BASE_DATASETS_DIR / "real1_cuant_categ_clean.csv"
    if clean_path.exists():
        clean_df = pd.read_csv(clean_path)
        feature_cols = [c for c in clean_df.columns if c != TARGET_COL]
        _CLEAN_DF_CACHE = clean_df.copy()
        _FEATURE_COLS_CACHE = list(feature_cols)
        return clean_df.copy(), list(feature_cols)

    if not DATA_FILE_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {DATA_FILE_NAME}. Ponlo en la misma carpeta que este script.")

    raw_df = pd.read_excel(DATA_FILE_PATH, sheet_name=EXCEL_SHEET_NAME)
    raw_df = _clean_column_names(raw_df)
    raw_df = raw_df.replace(r"^\s*$", np.nan, regex=True)
    raw_df = raw_df.drop(columns=[c for c in raw_df.columns if str(c).startswith("Unnamed")], errors="ignore")
    raw_df = raw_df.dropna(axis=1, how="all")
    raw_df = raw_df.dropna(axis=0, how="all").drop_duplicates().reset_index(drop=True)
    raw_df = raw_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    raw_df[TARGET_COL] = _coerce_binary_target(raw_df[TARGET_COL])
    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    quantitative_cols_initial = [c for c in numeric_cols if c != TARGET_COL]
    quantitative_cols = [c for c in quantitative_cols_initial if raw_df[c].nunique(dropna=True) > 1]

    categorical_candidate_cols = [
        c for c in raw_df.columns
        if c != TARGET_COL and c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(raw_df[c])
    ]
    categorical_cols = [
        c for c in categorical_candidate_cols
        if 2 <= raw_df[c].nunique(dropna=True) <= MAX_CATEGORICAL_UNIQUE_VALUES
    ]

    selected_original_cols = quantitative_cols + categorical_cols
    if not selected_original_cols:
        raise ValueError("No se han encontrado variables predictoras para entrenar el modelo.")

    clean_original_df = raw_df[selected_original_cols + [TARGET_COL]].copy()
    clean_original_df = clean_original_df.dropna(subset=selected_original_cols).reset_index(drop=True)
    if clean_original_df[TARGET_COL].nunique(dropna=True) < 2:
        raise ValueError("Tras la limpieza queda menos de dos clases en el target.")

    binary_categorical_cols = []
    multiclass_categorical_cols = []
    one_hot_order = {}

    for col in categorical_cols:
        clean_original_df[col] = clean_original_df[col].astype(str).str.strip()
        categories = sorted(clean_original_df[col].dropna().astype(str).str.strip().unique().tolist())
        if len(categories) == 2:
            binary_categorical_cols.append(col)
            clean_original_df[col] = clean_original_df[col].map({categories[0]: 0, categories[1]: 1}).astype(int)
        elif len(categories) > 2:
            multiclass_categorical_cols.append(col)
            reference = REFERENCE_CATEGORIES.get(col, categories[0])
            if reference not in categories:
                raise ValueError(f"La categoría de referencia {reference!r} no existe para {col}. Categorías: {categories}")
            encoded_categories = [cat for cat in categories if cat != reference]
            one_hot_order[col] = [reference] + encoded_categories

    if multiclass_categorical_cols:
        for col in multiclass_categorical_cols:
            clean_original_df[col] = pd.Categorical(clean_original_df[col], categories=one_hot_order[col], ordered=False)
        encoded_df = pd.get_dummies(
            clean_original_df,
            columns=multiclass_categorical_cols,
            prefix=multiclass_categorical_cols,
            prefix_sep="__",
            dummy_na=False,
            drop_first=True,
            dtype=int,
        )
    else:
        encoded_df = clean_original_df.copy()

    feature_cols = [c for c in encoded_df.columns if c != TARGET_COL]
    constant_cols = [c for c in feature_cols if encoded_df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        encoded_df = encoded_df.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    clean_df = encoded_df[feature_cols + [TARGET_COL]].copy()
    BASE_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_path, index=False)
    _CLEAN_DF_CACHE = clean_df.copy()
    _FEATURE_COLS_CACHE = list(feature_cols)
    return clean_df.copy(), list(feature_cols)


def get_feature_cols_from_df(df):
    excluded = {TARGET_COL, "split", "sample_id"}
    return [c for c in df.columns if c not in excluded]


def load_or_create_dataset(seed, scenario_name, config):
    clean_df, feature_cols = load_and_clean_real_dataset()
    scenario_dataset_dir = BASE_DATASETS_DIR / get_short_scenario_name(scenario_name)
    scenario_dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = scenario_dataset_dir / f"real1_cuant_categ_seed{seed:02d}.csv"
    required_cols = set(feature_cols + [TARGET_COL, "split", "sample_id"])

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        if required_cols.issubset(set(df.columns)):
            return df
        dataset_path.unlink()

    y = clean_df[TARGET_COL].to_numpy(dtype=int)
    indices = np.arange(len(clean_df))
    idx_train_val, idx_test, y_train_val, y_test = train_test_split(indices, y, test_size=0.10, random_state=seed, stratify=y)
    idx_train, idx_val, y_train, y_val = train_test_split(idx_train_val, y_train_val, test_size=0.20, random_state=seed, stratify=y_train_val)

    df_train = clean_df.iloc[idx_train].copy(); df_train["split"] = "train"; df_train["sample_id"] = idx_train
    df_val = clean_df.iloc[idx_val].copy(); df_val["split"] = "val"; df_val["sample_id"] = idx_val
    df_test = clean_df.iloc[idx_test].copy(); df_test["split"] = "test"; df_test["sample_id"] = idx_test
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df.to_csv(dataset_path, index=False)
    return df


# ============================================================
# 2. FUNCIONES COMUNES
# ============================================================

def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} min"
    hours = minutes / 60
    return f"{hours:.2f} h"


def sigmoid_stable(logits):
    logits = np.clip(np.asarray(logits, dtype=float), -709, 709)
    return 1.0 / (1.0 + np.exp(-logits))


def predict_from_saved_coefficients(X_scaled, intercept, betas):
    logits = X_scaled @ np.asarray(betas, dtype=float) + float(intercept)
    y_prob = sigmoid_stable(logits)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_prob, y_pred


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


def fit_model_with_convergence_info(model, X_train_scaled, y_train):
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

    return {
        "n_iter": n_iter,
        "max_iter": max_iter,
        "convergence_warning": convergence_warning,
        "reached_max_iter": reached_max_iter,
        "converged_without_warning": not convergence_warning,
        "convergence_message": " | ".join(convergence_warning_messages),
    }


def make_logistic_model(seed):
    # Para estrategias de remuestreo no se usa class_weight: el desbalanceo se trata en el TRAIN remuestreado.
    return LogisticRegression(random_state=seed, max_iter=MODEL_MAX_ITER)


def add_prediction_rows(predictions_list, scenario_name, seed, model_name, split_name, split_df, y_true, y_pred, y_prob, c0_value=1.0, c1_value=1.0):
    for sample_id, y_true_i, y_pred_i, y_prob_i in zip(split_df["sample_id"], y_true, y_pred, y_prob):
        predictions_list.append({
            "scenario": scenario_name,
            "seed": seed,
            "model_name": model_name,
            "split": split_name,
            "sample_id": int(sample_id),
            "c0": float(c0_value),
            "c1": float(c1_value),
            "y_true": int(y_true_i),
            "y_pred": int(y_pred_i),
            "y_prob": float(y_prob_i),
        })

STRATEGY_FOLDER_SHORT_NAMES = {"greedy_class_weight": "greedy", "random_oversampling": "oversampling", "smote": "smote", "random_undersampling": "undersampling"}

STRATEGY_FILE_SHORT_NAMES = {"greedy_class_weight": "greedy", "random_oversampling": "ros", "smote": "smote", "random_undersampling": "rus"}

def get_resampling_coefficients_path(scenario_name, strategy_name):
    scenario_short = get_short_scenario_name(scenario_name)

    strategy_folder_short = STRATEGY_FOLDER_SHORT_NAMES.get(strategy_name, strategy_name)
    strategy_file_short = STRATEGY_FILE_SHORT_NAMES.get(strategy_name, strategy_name)

    # Carpeta algo más corta, pero todavía legible
    strategy_dir = ensure_directory(COEFFICIENTS_DIR / strategy_folder_short)

    # Archivo CSV corto
    return strategy_dir / f"{strategy_file_short}_coef.csv"



def coefficients_are_compatible_for_strategy(scenario_name, strategy_name, feature_cols):
    path = get_resampling_coefficients_path(scenario_name, strategy_name)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    required = {"scenario", "seed", "model_name", "training_mode", "intercept", "n_iter", "max_iter"}
    if not required.issubset(set(df.columns)):
        return False
    if set(df["seed"].astype(int).unique()) != set(SEEDS):
        return False
    if set(df["model_name"].unique()) != {strategy_name}:
        return False
    if set(df["scenario"].unique()) != {scenario_name}:
        return False
    if set(df["training_mode"].astype(str).unique()) != {COMPARISON_RUN_SIGNATURE}:
        return False
    if not set(feature_cols).issubset(set(df.columns)):
        return False
    return True


def load_seed_resampling_coefficients(scenario_name, seed, strategy_name):
    path = get_resampling_coefficients_path(scenario_name, strategy_name)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["scenario"] == scenario_name) & (df["seed"].astype(int) == int(seed)) & (df["model_name"] == strategy_name)]
    if df.empty:
        return None
    return df.iloc[0]


def make_resampler(strategy_name, y_train, seed):
    y_train = np.asarray(y_train, dtype=int)
    class_counts = pd.Series(y_train).value_counts().to_dict()
    min_count = int(min(class_counts.values())) if class_counts else 0

    if strategy_name == "random_oversampling":
        return RandomOverSampler(random_state=seed), {"k_neighbors": np.nan}

    if strategy_name == "random_undersampling":
        return RandomUnderSampler(random_state=seed), {"k_neighbors": np.nan}

    if strategy_name == "smote":
        if min_count <= 1:
            raise ValueError(f"No hay suficientes muestras minoritarias para SMOTE: min_count={min_count}")
        k_neighbors = min(5, min_count - 1)
        return SMOTE(random_state=seed, k_neighbors=k_neighbors), {"k_neighbors": k_neighbors}

    raise ValueError(f"Estrategia de remuestreo no reconocida: {strategy_name}")



def _joblib_warmup_task(index):
    """
    Warm-up realista y acotado.

    No usa directamente RESAMPLING_MODEL_NAMES para evitar que, si en el futuro
    se añade una estrategia nueva con requisitos especiales, el warm-up intente
    ejecutarla automáticamente. Solo calienta las tres estrategias estándar de
    remuestreo usadas actualmente como pipeline genérico.
    """
    rng = np.random.default_rng(1000 + int(index))

    # Dataset pequeño, pero con ambas clases y desbalanceo suficiente para ejecutar
    # RandomOverSampler, SMOTE, RandomUnderSampler y LogisticRegression.
    X = rng.normal(size=(240, 12))
    y = np.array([0] * 192 + [1] * 48, dtype=int)
    order = rng.permutation(len(y))
    X = X[order]
    y = y[order]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for strategy_name in WARMUP_RESAMPLING_MODEL_NAMES:
        resampler, _ = make_resampler(strategy_name, y, int(index) + 1)
        X_resampled, y_resampled = resampler.fit_resample(X_scaled, y)
        model = LogisticRegression(random_state=int(index) + 1, max_iter=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_resampled, y_resampled)

    return index


def preload_datasets_for_timing():
    """
    Precarga los datasets/splits antes de medir.

    Esto evita que la primera estrategia medida cargue en solitario el coste de
    lectura de CSV/Excel, creación de splits o caché del sistema operativo.
    No se incluye en el coste comparable porque no pertenece a una estrategia
    concreta; es calentamiento técnico común.
    """
    t_start = time.perf_counter()
    n_loaded = 0

    try:
        for scenario_name, config in get_scenarios().items():
            for seed in SEEDS:
                df = load_or_create_dataset(seed, scenario_name, config)
                feature_cols = get_feature_cols_from_df(df)
                target_col = get_target_col()
                train_df = df[df["split"] == "train"].reset_index(drop=True)
                if not train_df.empty and feature_cols:
                    X_train = train_df[feature_cols].to_numpy(dtype=float)
                    y_train = train_df[target_col].to_numpy(dtype=int)
                    # Operación mínima para calentar conversión a numpy/escalado sin guardar nada.
                    _ = StandardScaler().fit_transform(X_train)
                    _ = np.bincount(y_train.astype(int), minlength=2)
                n_loaded += 1

        elapsed = time.perf_counter() - t_start
        print(f"Precarga técnica de datasets terminada: {n_loaded} splits/semillas · {format_seconds(elapsed)}. No se incluye en el coste comparable.")
    except Exception as exc:
        print(f"Precarga técnica de datasets omitida por error no crítico: {exc}")
        print("El script continúa, pero el primer bloque medido podría incluir algo de sobrecoste inicial.")


def run_pandas_numpy_metrics_warmup():
    """
    Warm-up técnico de pandas, NumPy/BLAS, métricas e imblearn.

    Inicializa operaciones usadas después en agregaciones, tablas, selección,
    predicción, métricas y remuestreo. Este tiempo no se asigna a ninguna
    estrategia y no se incorpora al coste comparable.
    """
    t_start = time.perf_counter()

    try:
        rng = np.random.default_rng(2026)
        scenarios = list(get_scenarios().keys()) or ["warmup"]
        n_rows = max(1800, len(SEEDS) * len(scenarios) * max(1, len(ALL_MODEL_NAMES)) * 6)

        warmup_df = pd.DataFrame({
            "scenario": rng.choice(scenarios, size=n_rows),
            "seed": rng.choice(SEEDS, size=n_rows),
            "model_name": rng.choice(ALL_MODEL_NAMES, size=n_rows),
            "c0": rng.choice([0.1, 0.5, 1.0, 5.0, 10.0, 100.0], size=n_rows),
            "c1": rng.choice([0.1, 0.5, 1.0, 5.0, 10.0, 100.0], size=n_rows),
            "val_f1": rng.random(n_rows),
            "val_pr_auc": rng.random(n_rows),
            "test_f1": rng.random(n_rows),
            "test_pr_auc": rng.random(n_rows),
        })

        _ = (
            warmup_df
            .groupby(["scenario", "model_name"])[["val_f1", "val_pr_auc", "test_f1", "test_pr_auc"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        _ = (
            warmup_df
            .groupby(["scenario", "seed", "c0", "c1"])[["val_f1", "test_f1"]]
            .mean()
            .reset_index()
        )
        _ = warmup_df.sort_values(["scenario", "model_name", "seed", "c0", "c1"]).reset_index(drop=True)
        _ = pd.concat([warmup_df.head(60), warmup_df.tail(60)], ignore_index=True)

        for scenario_name in scenarios:
            scenario_df = warmup_df[warmup_df["scenario"] == scenario_name]
            if not scenario_df.empty:
                _ = scenario_df["val_f1"].idxmax()

        X = rng.normal(size=(420, 48))
        beta = rng.normal(size=48)
        logits = X @ beta
        y_prob = sigmoid_stable(logits)
        y_true = np.array([0] * 300 + [1] * 120, dtype=int)
        y_true = y_true[rng.permutation(len(y_true))]
        y_pred = (y_prob >= np.median(y_prob)).astype(int)

        _ = np.mean(X, axis=0)
        _ = np.std(X, axis=0)
        _ = np.percentile(y_prob, [25, 50, 75])
        _ = np.bincount(y_true, minlength=2)
        _ = compute_metrics_for_split(y_true, y_pred, y_prob, "warmup")

        X_scaled = StandardScaler().fit_transform(X)
        for strategy_name in WARMUP_RESAMPLING_MODEL_NAMES:
            resampler, _ = make_resampler(strategy_name, y_true, 2026)
            X_resampled, y_resampled = resampler.fit_resample(X_scaled, y_true)
            _ = X_resampled.shape
            _ = np.bincount(y_resampled.astype(int), minlength=2)

        elapsed = time.perf_counter() - t_start
        print(f"Warm-up pandas/NumPy/métricas terminado: {format_seconds(elapsed)}. No se incluye en el coste comparable.")
    except Exception as exc:
        print(f"Warm-up pandas/NumPy/métricas omitido por error no crítico: {exc}")
        print("El script continúa, pero la primera agregación/evaluación podría incluir algo de sobrecoste inicial.")

def run_parallel_warmup():
    """
    Warm-up técnico realista previo a los bloques medidos.

    Se ejecuta antes de medir las estrategias nuevas para que el primer TRAIN
    real no pague costes únicos de lectura/caché, pandas, NumPy/BLAS, métricas,
    joblib/loky, imports internos de imblearn o inicialización de sklearn. Este
    tiempo no se incorpora al coste comparable de ninguna estrategia.
    """
    print("\nWarm-up técnico: precargando datasets e inicializando operaciones comunes.")
    preload_datasets_for_timing()
    run_pandas_numpy_metrics_warmup()

    n_tasks = max(1, min(len(SEEDS), 20))
    t_start = time.perf_counter()
    try:
        Parallel(n_jobs=N_JOBS)(delayed(_joblib_warmup_task)(i) for i in range(n_tasks))
        elapsed = time.perf_counter() - t_start
        print(f"Warm-up joblib/sklearn/imblearn terminado: {format_seconds(elapsed)}. No se incluye en el coste comparable.")
    except Exception as exc:
        print(f"Warm-up joblib/sklearn/imblearn omitido por error no crítico: {exc}")
        print("El script continúa, pero el primer bloque medido podría incluir algo de sobrecoste inicial.")

def train_one_seed_greedy_strategy(seed, scenario_name, config):
    """
    TRAIN REAL · Una seed completa para la heurística greedy de pesos de clase.

    La búsqueda de pesos se realiza únicamente sobre TRAIN mediante CV interna
    estratificada de 5 folds. Una vez seleccionados los pesos, la propia clase
    realiza refit sobre todo TRAIN con refit=True.
    """
    t_start = time.perf_counter()

    df = load_or_create_dataset(seed, scenario_name, config)
    feature_cols = get_feature_cols_from_df(df)
    target_col = get_target_col()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    greedy = GreedyClassWeightLogisticRegressionCV(
        weight_bounds=(0.01, 10),
        cv=StratifiedKFold(n_splits=5),
        scoring=make_scorer(f1_score, zero_division=0),
        max_iter=500,
        random_state=seed,
        n_jobs=1,
        refit=True,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        greedy.fit(X_train_scaled, y_train)

    convergence_warning_messages = [
        str(w.message)
        for w in caught_warnings
        if issubclass(w.category, ConvergenceWarning)
    ]

    final_estimator = _get_greedy_final_estimator(greedy)
    best_class_weight = getattr(greedy, "best_class_weight_", {})
    c0_value = _extract_class_weight_value(best_class_weight, 0)
    c1_value = _extract_class_weight_value(best_class_weight, 1)

    n_iter_value = np.nan
    if hasattr(final_estimator, "n_iter_"):
        try:
            n_iter_value = int(np.max(final_estimator.n_iter_))
        except Exception:
            n_iter_value = np.nan

    max_iter_value = int(MODEL_MAX_ITER)
    reached_max_iter = bool(n_iter_value >= max_iter_value) if not pd.isna(n_iter_value) else False
    convergence_warning = len(convergence_warning_messages) > 0

    coeff_row = {
        "scenario": scenario_name,
        "seed": seed,
        "model_name": "greedy_class_weight",
        "training_mode": COMPARISON_RUN_SIGNATURE,
        "c0": c0_value,
        "c1": c1_value,
        "greedy_best_score_cv": float(getattr(greedy, "best_score_", np.nan)),
        "greedy_weight_pairs_evaluated": _unique_weight_evaluations(getattr(greedy, "cv_results_", [])),
        "greedy_cv_folds": 5,
        "greedy_n_rounds": getattr(greedy, "n_rounds_", np.nan),
        "intercept": float(final_estimator.intercept_[0]),
        "n_iter": n_iter_value,
        "max_iter": max_iter_value,
        "convergence_warning": convergence_warning,
        "reached_max_iter": reached_max_iter,
        "converged_without_warning": not convergence_warning,
        "convergence_message": " | ".join(convergence_warning_messages),
    }
    for feature_name, beta in zip(feature_cols, final_estimator.coef_[0]):
        coeff_row[feature_name] = float(beta)

    total_seconds = time.perf_counter() - t_start

    greedy_pairs = coeff_row["greedy_weight_pairs_evaluated"]
    greedy_folds = 5

    if pd.isna(greedy_pairs):
        greedy_cv_models = np.nan
        greedy_total_models = np.nan
    else:
        greedy_cv_models = int(greedy_pairs) * int(greedy_folds)
        greedy_total_models = greedy_cv_models + 1  # refit final sobre todo TRAIN

    timer_row = {
        "scenario": scenario_name,
        "seed": seed,
        "stage": "train_greedy_class_weight_seed",
        "model_name": "greedy_class_weight",
        "n_models": greedy_total_models,
        "greedy_weight_pairs_evaluated": coeff_row["greedy_weight_pairs_evaluated"],
        "greedy_cv_folds": greedy_folds,
        "greedy_cv_models": greedy_cv_models,
        "greedy_refit_models": 1,
        "total_seconds": float(total_seconds),
        "total_time_readable": format_seconds(total_seconds),
        "skipped_because_existing_outputs": False,
    }

    return pd.DataFrame([coeff_row]), pd.DataFrame([timer_row])


def train_or_load_greedy_strategy(scenario_name, config, timer_frames):
    sample_df = load_or_create_dataset(SEEDS[0], scenario_name, config)
    feature_cols = get_feature_cols_from_df(sample_df)

    if coefficients_are_compatible_for_strategy(scenario_name, "greedy_class_weight", feature_cols):
        print(f"Coeficientes existentes compatibles: {scenario_name} · greedy_class_weight. Se reutilizan.")
        timer_frames.append(pd.DataFrame([{
            "scenario": scenario_name,
            "seed": "all",
            "stage": "load_existing_greedy_class_weight_coefficients",
            "model_name": "greedy_class_weight",
            "n_models": 0,
            "total_seconds": np.nan,
            "total_time_readable": "coeficientes reutilizados",
            "skipped_because_existing_outputs": True,
        }]))
        return

    print("\n" + "=" * 90)
    print(f"TRAIN REAL · GREEDY CLASS WEIGHT · ESCENARIO: {scenario_name.upper()}")
    print("=" * 90)
    print(f"Ejecutando {len(SEEDS)} seeds en paralelo.")
    print("TRAIN real incluye: carga/preparación del TRAIN + escalado + CV interna greedy 5 folds + refit LogisticRegression.")
    print("La heurística greedy solo usa TRAIN. VALIDACIÓN y TEST quedan para evaluación.")
    print("Paralelización: solo seeds externas. Greedy interno con n_jobs=1.")

    train_wall_start = time.perf_counter()
    parallel_output = Parallel(n_jobs=N_JOBS)(
        delayed(train_one_seed_greedy_strategy)(seed, scenario_name, config)
        for seed in SEEDS
    )
    train_wall_seconds = time.perf_counter() - train_wall_start

    coeff_df = pd.concat([item[0] for item in parallel_output], ignore_index=True)
    timer_df = pd.concat([item[1] for item in parallel_output], ignore_index=True)

    coeff_df = coeff_df.sort_values(["scenario", "seed", "model_name"]).reset_index(drop=True)
    coeff_df.to_csv(get_resampling_coefficients_path(scenario_name, "greedy_class_weight"), index=False)

    timer_frames.append(timer_df)

    n_greedy_models_total = int(pd.to_numeric(
        timer_df["n_models"],
        errors="coerce",
    ).fillna(0.0).sum())

    timer_frames.append(pd.DataFrame([{
        "scenario": scenario_name,
        "seed": "all",
        "stage": "greedy_class_weight_train_parallel_total",
        "model_name": "greedy_class_weight",
        "n_models": n_greedy_models_total,
        "total_seconds": float(train_wall_seconds),
        "total_time_readable": format_seconds(train_wall_seconds),
        "skipped_because_existing_outputs": False,
    }]))

    print(f"Tiempo real TRAIN: {format_seconds(train_wall_seconds)}")
    print(f"Coeficientes guardados: {get_resampling_coefficients_path(scenario_name, 'greedy_class_weight')}")


def train_one_seed_resampling_strategy(seed, scenario_name, config, strategy_name):
    """
    TRAIN REAL · Una seed completa para una estrategia de remuestreo.

    Incluye en un único flujo:
    - cargar el dataset/split ya generado por el código base;
    - ajustar StandardScaler solo con TRAIN;
    - aplicar el remuestreo únicamente sobre TRAIN;
    - entrenar LogisticRegression estándar sin class_weight;
    - devolver coeficientes wide para reutilizarlos después.

    Esta función evita separar artificialmente remuestreo y entrenamiento en
    bloques distintos. El tiempo comparable se mide fuera, como tiempo real
    del bloque paralelo completo de las 20 seeds.
    """
    t_start = time.perf_counter()

    df = load_or_create_dataset(seed, scenario_name, config)
    feature_cols = get_feature_cols_from_df(df)
    target_col = get_target_col()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    resampler, _ = make_resampler(strategy_name, y_train, seed)
    X_resampled, y_resampled = resampler.fit_resample(X_train_scaled, y_train)

    model = make_logistic_model(seed)
    convergence_info = fit_model_with_convergence_info(model, X_resampled, y_resampled)

    coeff_row = {
        "scenario": scenario_name,
        "seed": seed,
        "model_name": strategy_name,
        "training_mode": COMPARISON_RUN_SIGNATURE,
        "c0": 1.0,
        "c1": 1.0,
        "intercept": float(model.intercept_[0]),
        "n_iter": convergence_info["n_iter"],
        "max_iter": convergence_info["max_iter"],
        "convergence_warning": convergence_info["convergence_warning"],
        "reached_max_iter": convergence_info["reached_max_iter"],
        "converged_without_warning": convergence_info["converged_without_warning"],
        "convergence_message": convergence_info["convergence_message"],
    }
    for feature_name, beta in zip(feature_cols, model.coef_[0]):
        coeff_row[feature_name] = float(beta)

    total_seconds = time.perf_counter() - t_start
    timer_row = {
        "scenario": scenario_name,
        "seed": seed,
        "stage": "train_resampling_strategy_seed",
        "model_name": strategy_name,
        "n_models": 1,
        "total_seconds": float(total_seconds),
        "total_time_readable": format_seconds(total_seconds),
        "skipped_because_existing_outputs": False,
    }

    return pd.DataFrame([coeff_row]), pd.DataFrame([timer_row])

def train_or_load_resampling_strategy(scenario_name, config, strategy_name, timer_frames):
    sample_df = load_or_create_dataset(SEEDS[0], scenario_name, config)
    feature_cols = get_feature_cols_from_df(sample_df)

    if coefficients_are_compatible_for_strategy(scenario_name, strategy_name, feature_cols):
        print(f"Coeficientes existentes compatibles: {scenario_name} · {strategy_name}. Se reutilizan.")
        timer_frames.append(pd.DataFrame([{
            "scenario": scenario_name,
            "seed": "all",
            "stage": "load_existing_resampling_coefficients",
            "model_name": strategy_name,
            "n_models": 0,
            "total_seconds": np.nan,
            "total_time_readable": "coeficientes reutilizados",
            "skipped_because_existing_outputs": True,
        }]))
        return

    print("\n" + "=" * 90)
    print(f"TRAIN REAL · {strategy_name.upper()} · ESCENARIO: {scenario_name.upper()}")
    print("=" * 90)
    print(f"Ejecutando {len(SEEDS)} seeds en paralelo.")
    print("TRAIN real incluye: carga/preparación del TRAIN + escalado + remuestreo en TRAIN + entrenamiento LogisticRegression.")
    print("VALIDACIÓN y TEST no se remuestrean.")

    train_wall_start = time.perf_counter()
    parallel_output = Parallel(n_jobs=N_JOBS)(
        delayed(train_one_seed_resampling_strategy)(seed, scenario_name, config, strategy_name)
        for seed in SEEDS
    )
    train_wall_seconds = time.perf_counter() - train_wall_start

    coeff_df = pd.concat([item[0] for item in parallel_output], ignore_index=True)
    timer_df = pd.concat([item[1] for item in parallel_output], ignore_index=True)

    coeff_df = coeff_df.sort_values(["scenario", "seed", "model_name"]).reset_index(drop=True)
    coeff_df.to_csv(get_resampling_coefficients_path(scenario_name, strategy_name), index=False)

    timer_frames.append(timer_df)
    timer_frames.append(pd.DataFrame([{
        "scenario": scenario_name,
        "seed": "all",
        "stage": f"{strategy_name}_train_parallel_total",
        "model_name": strategy_name,
        "n_models": len(SEEDS),
        "total_seconds": float(train_wall_seconds),
        "total_time_readable": format_seconds(train_wall_seconds),
        "skipped_because_existing_outputs": False,
    }]))

    print(f"Tiempo real TRAIN: {format_seconds(train_wall_seconds)}")
    print(f"Coeficientes guardados: {get_resampling_coefficients_path(scenario_name, strategy_name)}")


def evaluate_one_seed_resampling_strategy(seed, scenario_name, config, strategy_name, split_name):
    t_start = time.perf_counter()

    df = load_or_create_dataset(seed, scenario_name, config)
    feature_cols = get_feature_cols_from_df(df)
    target_col = get_target_col()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    eval_df = df[df["split"] == split_name].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_eval = eval_df[feature_cols].to_numpy(dtype=float)
    y_eval = eval_df[target_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    coeff_row = load_seed_resampling_coefficients(scenario_name, seed, strategy_name)
    if coeff_row is None:
        raise FileNotFoundError(
            f"Faltan coeficientes para scenario={scenario_name}, seed={seed}, strategy={strategy_name}. "
            "Ejecuta primero la fase TRAIN de estrategias de remuestreo."
        )

    intercept = float(coeff_row["intercept"])
    betas = coeff_row[feature_cols].to_numpy(dtype=float)

    t_pred_start = time.perf_counter()
    y_prob, y_pred = predict_from_saved_coefficients(X_eval_scaled, intercept, betas)
    prediction_seconds = time.perf_counter() - t_pred_start

    t_metrics_start = time.perf_counter()
    metrics = compute_metrics_for_split(y_eval, y_pred, y_prob, split_name)
    metrics_seconds = time.perf_counter() - t_metrics_start

    c0_value = float(coeff_row.get("c0", 1.0)) if not pd.isna(coeff_row.get("c0", 1.0)) else 1.0
    c1_value = float(coeff_row.get("c1", 1.0)) if not pd.isna(coeff_row.get("c1", 1.0)) else 1.0

    result_row = {
        "scenario": scenario_name,
        "seed": seed,
        "model_name": strategy_name,
        "c0": c0_value,
        "c1": c1_value,
        "n_iter": coeff_row.get("n_iter", np.nan),
        "max_iter": coeff_row.get("max_iter", np.nan),
        "convergence_warning": bool(coeff_row.get("convergence_warning", False)),
        "reached_max_iter": bool(coeff_row.get("reached_max_iter", False)),
        "converged_without_warning": coeff_row.get("converged_without_warning", np.nan),
        "convergence_message": coeff_row.get("convergence_message", ""),
        "used_saved_coefficients": True,
    }
    for optional_col in ["greedy_best_score_cv", "greedy_weight_pairs_evaluated", "greedy_cv_folds", "greedy_n_rounds"]:
        if optional_col in coeff_row.index:
            result_row[optional_col] = coeff_row.get(optional_col, np.nan)
    result_row.update(metrics)

    predictions = []
    add_prediction_rows(predictions, scenario_name, seed, strategy_name, split_name, eval_df, y_eval, y_pred, y_prob, c0_value=c0_value, c1_value=c1_value)

    timer_row = {
        "scenario": scenario_name,
        "seed": seed,
        "stage": f"evaluate_{split_name}_resampling_strategy",
        "model_name": strategy_name,
        "n_models": 1,
        "prediction_seconds": prediction_seconds,
        "metrics_seconds": metrics_seconds,
        "total_seconds": time.perf_counter() - t_start,
        "total_time_readable": format_seconds(time.perf_counter() - t_start),
        "skipped_because_existing_outputs": False,
    }

    return pd.DataFrame([result_row]), pd.DataFrame(predictions), pd.DataFrame([timer_row])


def evaluate_resampling_strategy(scenario_name, config, strategy_name, split_name, timer_frames):
    print(f"Evaluando {strategy_name} en {split_name.upper()} · {scenario_name}")
    wall_start = time.perf_counter()
    parallel_output = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_one_seed_resampling_strategy)(seed, scenario_name, config, strategy_name, split_name)
        for seed in SEEDS
    )

    results_df = pd.concat([item[0] for item in parallel_output], ignore_index=True)
    predictions_df = pd.concat([item[1] for item in parallel_output], ignore_index=True)
    timer_df = pd.concat([item[2] for item in parallel_output], ignore_index=True)
    timer_frames.append(timer_df)
    timer_frames.append(pd.DataFrame([{
        "scenario": scenario_name,
        "seed": "all",
        "stage": f"{strategy_name}_{split_name}_parallel_total",
        "model_name": strategy_name,
        "n_models": len(SEEDS),
        "prediction_seconds": np.nan,
        "metrics_seconds": np.nan,
        "total_seconds": time.perf_counter() - wall_start,
        "total_time_readable": format_seconds(time.perf_counter() - wall_start),
        "skipped_because_existing_outputs": False,
    }]))

    return results_df, predictions_df


def load_base_results_and_predictions(split_name):
    if split_name == "val":
        results_path = BASE_VAL_RESULTS_PATH
        predictions_path = BASE_VAL_PREDICTIONS_PATH
    elif split_name == "test":
        results_path = BASE_TEST_RESULTS_PATH
        predictions_path = BASE_TEST_PREDICTIONS_PATH
    else:
        raise ValueError(split_name)

    if not results_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            f"No encuentro los resultados base WBCE para {split_name.upper()}.\n"
            f"Falta: {results_path if not results_path.exists() else predictions_path}\n"
            "Ejecuta primero el código base correspondiente para generar la rejilla WBCE, Equal weights y Sklearn balanced."
        )

    base_results = pd.read_csv(results_path)
    base_predictions = pd.read_csv(predictions_path)

    base_results = base_results[base_results["model_name"].isin(BASE_MODEL_NAMES)].copy()
    base_predictions = base_predictions[base_predictions["model_name"].isin(BASE_MODEL_NAMES)].copy()

    if "split" not in base_predictions.columns:
        base_predictions["split"] = split_name

    return base_results, base_predictions


def aggregate_results(raw_df, group_cols):
    metric_cols = [
        col for col in raw_df.columns
        if col not in ["scenario", "seed", "model_name", "convergence_message"]
        and pd.api.types.is_numeric_dtype(raw_df[col])
    ]

    agg_df = raw_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()

    new_columns = []
    for col in agg_df.columns:
        if col[1] == "":
            new_columns.append(col[0])
        else:
            new_columns.append(f"{col[0]}_{col[1]}")
    agg_df.columns = new_columns

    std_cols = [col for col in agg_df.columns if col.endswith("_std")]
    agg_df[std_cols] = agg_df[std_cols].fillna(0.0)

    model_rank = {name: idx for idx, name in enumerate(ALL_MODEL_NAMES)}
    agg_df["_model_rank"] = agg_df["model_name"].map(model_rank).fillna(999)
    agg_df = agg_df.sort_values(["scenario", "_model_rank"]).drop(columns="_model_rank").reset_index(drop=True)
    return agg_df


def make_comparison_table(agg_df, split_name):
    cols = [
        "scenario", "model_name", "c0_mean", "c0_std", "c1_mean", "c1_std",
        f"{split_name}_f1_mean", f"{split_name}_f1_std",
        f"{split_name}_pr_auc_mean", f"{split_name}_pr_auc_std",
        f"{split_name}_balanced_accuracy_mean", f"{split_name}_balanced_accuracy_std",
        f"{split_name}_mcc_mean", f"{split_name}_mcc_std",
        f"{split_name}_recall_mean", f"{split_name}_recall_std",
        f"{split_name}_fnr_mean", f"{split_name}_fnr_std",
        f"{split_name}_specificity_mean", f"{split_name}_specificity_std",
        f"{split_name}_roc_auc_mean", f"{split_name}_roc_auc_std",
    ]
    existing_cols = [c for c in cols if c in agg_df.columns]
    return agg_df[existing_cols].copy()


def format_split_label_for_plot(split_name):
    split_name = str(split_name).lower()
    labels = {
        "train": "TRAIN",
        "val": "VALIDACIÓN",
        "test": "TEST",
    }
    return labels.get(split_name, split_name.upper())


def format_stat_label_for_plot(stat_name):
    stat_name = str(stat_name).lower()
    labels = {
        "mean": "Mean",
        "std": "STD",
    }
    return labels.get(stat_name, stat_name.upper())


def format_metric_name_for_plot(metric_name):
    metric_name = str(metric_name).lower()
    labels = {
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
        "pr_auc": "PR AUC",
        "roc_auc": "ROC AUC",
    }

    return labels.get(metric_name, metric_name.replace("_", " ").title())


def format_metric_label_for_plot(metric, include_split=True):
    parts = str(metric).split("_")
    split = None
    stat = None

    if parts and parts[0] in {"train", "val", "test"}:
        split = parts[0]
        parts = parts[1:]

    if parts and parts[-1] in {"mean", "std"}:
        stat = parts[-1]
        parts = parts[:-1]

    metric_name = "_".join(parts)
    label_parts = []

    if include_split and split is not None:
        label_parts.append(format_split_label_for_plot(split))

    label_parts.append(format_metric_name_for_plot(metric_name))

    if stat is not None:
        label_parts.append(format_stat_label_for_plot(stat))

    return " · ".join(label_parts)


def format_scenario_label_for_plot(scenario_name):
    labels = {
        "real_1_cuantitativas_categoricas": "Real 1 Cuantitativas + Categóricas",
    }
    scenario_name = str(scenario_name)
    return labels.get(scenario_name, scenario_name.replace("_", " ").title())


def format_weight_value_for_plot(value):
    """Formatea pesos de forma compacta para las etiquetas de las barras."""
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def make_bar_labels(metric_mean, metric_std, c0_value, c1_value, model_name=None, c0_std=np.nan, c1_std=np.nan):
    """
    Etiquetas compactas para cada barra de las figuras comparativas.

    Se separan en dos columnas verticales:
    - una columna para mean y std;
    - otra columna para c0 y c1.

    En la heurística greedy también se muestra la desviación típica de c0 y c1,
    ya que es la única estrategia donde esos pesos varían entre seeds.
    """
    if pd.isna(metric_mean):
        return "", ""

    c0_text = format_weight_value_for_plot(c0_value)
    c1_text = format_weight_value_for_plot(c1_value)
    std_value = 0.0 if pd.isna(metric_std) else float(metric_std)

    metric_text = f"Mean={float(metric_mean):.3f}\nSTD={std_value:.3f}"

    if model_name == "greedy_class_weight":
        c0_std_text = format_weight_value_for_plot(c0_std)
        c1_std_text = format_weight_value_for_plot(c1_std)
        weights_text = f"c0={c0_text} ± {c0_std_text}\nc1={c1_text} ± {c1_std_text}"
    else:
        weights_text = f"c0={c0_text}\nc1={c1_text}"

    return metric_text, weights_text


def save_bar_comparison_plots(agg_df, split_name):
    metrics = [
        f"{split_name}_f1_mean",
        f"{split_name}_pr_auc_mean",
        f"{split_name}_balanced_accuracy_mean",
        f"{split_name}_mcc_mean",
        f"{split_name}_recall_mean",
        f"{split_name}_fnr_mean",
        f"{split_name}_specificity_mean",
        f"{split_name}_roc_auc_mean",
    ]

    scenario_order = [s for s in get_scenarios().keys() if s in set(agg_df["scenario"].unique())]
    model_order = [m for m in PREFERRED_PLOT_ORDER if m in set(agg_df["model_name"].unique())]

    for metric in metrics:
        if metric not in agg_df.columns:
            continue
        metric_std = metric.replace("_mean", "_std")

        x = np.arange(len(scenario_order))
        width = min(0.11, 0.78 / max(1, len(model_order)))
        fig, ax = plt.subplots(figsize=(max(13.8, 3.2 * len(scenario_order)), 8.6))

        all_values = []
        all_upper_values = []
        all_lower_values = []
        pending_labels = []

        for idx, model_name in enumerate(model_order):
            offset = (idx - (len(model_order) - 1) / 2.0) * width
            values, errors, bar_labels = [], [], []

            for scenario in scenario_order:
                row = agg_df[(agg_df["scenario"] == scenario) & (agg_df["model_name"] == model_name)]
                if row.empty:
                    values.append(np.nan)
                    errors.append(0.0)
                    bar_labels.append("")
                else:
                    r = row.iloc[0]
                    value = float(r[metric])
                    error = float(r[metric_std]) if metric_std in agg_df.columns and not pd.isna(r[metric_std]) else 0.0
                    c0_value = float(r["c0_mean"]) if "c0_mean" in agg_df.columns and not pd.isna(r.get("c0_mean", np.nan)) else np.nan
                    c1_value = float(r["c1_mean"]) if "c1_mean" in agg_df.columns and not pd.isna(r.get("c1_mean", np.nan)) else np.nan
                    c0_std_value = float(r["c0_std"]) if "c0_std" in agg_df.columns and not pd.isna(r.get("c0_std", np.nan)) else np.nan
                    c1_std_value = float(r["c1_std"]) if "c1_std" in agg_df.columns and not pd.isna(r.get("c1_std", np.nan)) else np.nan

                    values.append(value)
                    errors.append(error)
                    bar_labels.append(make_bar_labels(value, error, c0_value, c1_value, model_name=model_name, c0_std=c0_std_value, c1_std=c1_std_value))

            values = np.asarray(values, dtype=float)
            errors = np.asarray(errors, dtype=float)
            all_values.extend(values[~np.isnan(values)].tolist())
            all_upper_values.extend((values[~np.isnan(values)] + errors[~np.isnan(values)]).tolist())
            all_lower_values.extend((values[~np.isnan(values)] - errors[~np.isnan(values)]).tolist())

            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=MODEL_LABELS.get(model_name, model_name),
                color=MODEL_COLORS.get(model_name, "#777777"),
                edgecolor="white",
                linewidth=0.6,
                yerr=errors,
                capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
            )

            for bar, label_pair, value, error in zip(bars, bar_labels, values, errors):
                metric_text, weights_text = label_pair if isinstance(label_pair, tuple) else ("", "")
                if (not metric_text and not weights_text) or pd.isna(value):
                    continue
                pending_labels.append({
                    "bar": bar,
                    "metric_text": metric_text,
                    "weights_text": weights_text,
                    "value": float(value),
                    "error": float(error),
                    "model_name": model_name,
                })

        ax.set_xticks(x)
        ax.set_xticklabels([format_scenario_label_for_plot(s) for s in scenario_order], rotation=0, ha="center")
        ax.set_xlabel("Escenario", labelpad=10)
        ax.set_ylabel(format_metric_label_for_plot(metric, include_split=True))
        ax.set_title(
            f"Comparación de estrategias · {format_split_label_for_plot(split_name)} · {format_metric_label_for_plot(metric, include_split=False)}",
            fontsize=13,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        if all_values:
            ymin = min(0.0, min(all_lower_values) - 0.10)
            ymax = max(1.0, max(all_upper_values) + 0.42)
            if metric.endswith("_mcc_mean"):
                ymin = min(-0.10, min(all_lower_values) - 0.16)
                ymax = max(0.30, max(all_upper_values) + 0.42)
            ax.set_ylim(ymin, ymax)

        y_min_current, y_max_current = ax.get_ylim()
        y_span = y_max_current - y_min_current
        label_offset = 0.018 * y_span
        block_gap = 0.120 * y_span
        max_used_y = y_max_current
        min_used_y = y_min_current

        for item in pending_labels:
            bar = item["bar"]
            value = item["value"]
            error = item["error"]
            x_center = bar.get_x() + bar.get_width() / 2

            if item.get("model_name") == "greedy_class_weight":
                item_label_offset = 0.030 * y_span
                item_block_gap = 0.210 * y_span
            else:
                item_label_offset = label_offset
                item_block_gap = block_gap

            if value >= 0:
                y_weights = value + error + item_label_offset
                y_metrics = y_weights + item_block_gap
                va = "bottom"
                max_used_y = max(max_used_y, y_metrics)
            else:
                y_metrics = value - error - item_label_offset
                y_weights = y_metrics - item_block_gap
                va = "top"
                min_used_y = min(min_used_y, y_weights)

            ax.text(
                x_center,
                y_metrics,
                item["metric_text"],
                ha="center",
                va=va,
                fontsize=4.8,
                rotation=90,
                color="#111111",
                clip_on=False,
            )

            ax.text(
                x_center,
                y_weights,
                item["weights_text"],
                ha="center",
                va=va,
                fontsize=4.8,
                rotation=90,
                color="#111111",
                clip_on=False,
            )

        extra_top_margin = 0.30 * y_span
        extra_bottom_margin = 0.08 * y_span
        if max_used_y + extra_top_margin > y_max_current or min_used_y - extra_bottom_margin < y_min_current:
            ax.set_ylim(
                min(y_min_current, min_used_y - extra_bottom_margin),
                max(y_max_current, max_used_y + extra_top_margin),
            )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=8.5, frameon=True)
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        fig.savefig(FIGURES_DIR / f"{split_name}_comparison_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _draw_confusion_matrix_axis(ax, group, model_name, split_name):
    """Dibuja una matriz de confusión agregada en un eje concreto."""
    y_true = group["y_true"].to_numpy(dtype=int)
    y_pred = group["y_pred"].to_numpy(dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8.5)
    ax.set_yticklabels(["Real 0", "Real 1"], fontsize=8.5)
    ax.set_xlabel("Predicción", fontsize=8.5)
    ax.set_ylabel("Clase real", fontsize=8.5)

    n_seeds = group["seed"].nunique() if "seed" in group.columns else np.nan
    ax.set_title(
        f"{MODEL_LABELS.get(model_name, model_name)}\n{format_split_label_for_plot(split_name)} · {n_seeds} seeds · n={len(group)}",
        fontsize=9.5,
        pad=7,
    )

    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_pct[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def save_confusion_matrix_panels(predictions_df, split_name):
    """
    Guarda las matrices de confusión con Equal weights separado como referencia
    y las seis estrategias de tratamiento del desbalanceo en un panel 2x3.

    No modifica entrenamiento, evaluación, métricas ni predicciones. Solo cambia
    la forma de representar las matrices de confusión ya calculadas.
    """
    predictions_df = predictions_df.copy()
    if "split" in predictions_df.columns:
        predictions_df = predictions_df[predictions_df["split"] == split_name].copy()

    if predictions_df.empty:
        return

    scenario_order = [
        s
        for s in get_scenarios().keys()
        if s in set(predictions_df["scenario"].unique())
    ]
    model_order = [m for m in ALL_MODEL_NAMES if m in set(predictions_df["model_name"].unique())]

    for scenario in scenario_order:
        scenario_df = predictions_df[predictions_df["scenario"] == scenario].copy()
        if scenario_df.empty:
            continue

        # 1) Equal weights se guarda aparte como referencia 1:1.
        if "equal_weights" in model_order:
            equal_df = scenario_df[scenario_df["model_name"] == "equal_weights"].copy()
            if not equal_df.empty:
                fig, ax = plt.subplots(figsize=(5.8, 5.3))
                im = _draw_confusion_matrix_axis(ax, equal_df, "equal_weights", split_name)
                fig.suptitle(
                    f"{format_split_label_for_plot(split_name)} · Matriz de confusión referencia 1:1 · {format_scenario_label_for_plot(scenario)}",
                    fontsize=13.5,
                    y=0.985,
                )
                fig.subplots_adjust(left=0.13, right=0.84, bottom=0.10, top=0.84)
                cbar_ax = fig.add_axes([0.875, 0.19, 0.030, 0.58])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label("Porcentaje por clase real", rotation=90, labelpad=10)
                fig.savefig(
                    FIGURES_DIR / f"{split_name}_confusion_matrix_equal_weights_{get_short_scenario_name(scenario)}.png",
                    dpi=180,
                    bbox_inches="tight",
                    pad_inches=0.12,
                )
                plt.close(fig)

        # 2) Panel principal: las seis estrategias de desbalanceo.
        imbalance_order = [
            "best_val_f1_grid",
            "greedy_class_weight",
            "sklearn_balanced",
            "random_oversampling",
            "random_undersampling",
            "smote",
        ]
        imbalance_order = [m for m in imbalance_order if m in model_order]

        if not imbalance_order:
            continue

        ncols = 3
        nrows = int(np.ceil(len(imbalance_order) / ncols))
        nrows = max(1, nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.75 * nrows + 0.8))
        axes = np.asarray(axes).reshape(-1)
        last_im = None

        for ax, model_name in zip(axes, imbalance_order):
            group = scenario_df[scenario_df["model_name"] == model_name].copy()
            if group.empty:
                ax.axis("off")
                continue
            last_im = _draw_confusion_matrix_axis(ax, group, model_name, split_name)

        for ax in axes[len(imbalance_order):]:
            ax.axis("off")

        fig.suptitle(
            f"{format_split_label_for_plot(split_name)} · Matrices de confusión · Estrategias de desbalanceo · {format_scenario_label_for_plot(scenario)}",
            fontsize=15,
            y=0.985,
        )
        fig.subplots_adjust(left=0.055, right=0.90, bottom=0.070, top=0.900, wspace=0.28, hspace=0.46)
        if last_im is not None:
            cbar_ax = fig.add_axes([0.925, 0.20, 0.018, 0.55])
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label("Porcentaje por clase real", rotation=90, labelpad=10)
        fig.savefig(
            FIGURES_DIR / f"{split_name}_confusion_matrix_desbalanceo_panel_{get_short_scenario_name(scenario)}.png",
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.12,
        )
        plt.close(fig)

def save_boxplot_plots(raw_df, split_name):
    metrics = [
        f"{split_name}_f1",
        f"{split_name}_pr_auc",
        f"{split_name}_balanced_accuracy",
        f"{split_name}_mcc",
        f"{split_name}_recall",
        f"{split_name}_fnr",
        f"{split_name}_specificity",
        f"{split_name}_roc_auc",
    ]
    scenario_order = [s for s in get_scenarios().keys() if s in set(raw_df["scenario"].unique())]
    model_order = [m for m in PREFERRED_PLOT_ORDER if m in set(raw_df["model_name"].unique())]
    summary_rows = []

    for metric in metrics:
        if metric not in raw_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(15.0, 3.4 * len(scenario_order)), 8.2))
        box_data, positions, colors = [], [], []
        group_width = len(model_order) + 1
        xticks, xticklabels = [], []
        for s_idx, scenario in enumerate(scenario_order):
            base = s_idx * group_width
            xticks.append(base + (len(model_order) - 1) / 2.0)
            xticklabels.append(format_scenario_label_for_plot(scenario))
            for m_idx, model_name in enumerate(model_order):
                values = raw_df[(raw_df["scenario"] == scenario) & (raw_df["model_name"] == model_name)][metric].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                box_data.append(values)
                positions.append(base + m_idx)
                colors.append(MODEL_COLORS.get(model_name, "#777777"))
                summary_rows.append({
                    "split": split_name,
                    "metric": metric,
                    "scenario": scenario,
                    "model_name": model_name,
                    "n_seeds": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q1": float(np.percentile(values, 25)),
                    "q3": float(np.percentile(values, 75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                })
        if not box_data:
            plt.close(fig)
            continue
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="#111111", markersize=5.0),
            flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="#111111", markersize=3.5, linestyle="none"),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#222222")
            patch.set_linewidth(0.8)
        for median in bp["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, ha="center")
        ax.set_xlabel("Escenario", labelpad=10)
        ax.set_ylabel(format_metric_label_for_plot(metric, include_split=True))
        ax.set_title(
            f"Caja y bigotes · {format_split_label_for_plot(split_name)} · {format_metric_label_for_plot(metric, include_split=False)}",
            fontsize=13,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        handles = [Line2D([0], [0], color=MODEL_COLORS.get(m, "#777777"), linewidth=6, label=MODEL_LABELS.get(m, m)) for m in model_order]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5, frameon=True)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(FIGURES_DIR / f"{split_name}_boxplot_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / f"{split_name}_boxplot_table.csv", index=False)
    return summary_df




def save_pr_roc_curve_panels(predictions_df, results_df, split_name):
    """
    Guarda curvas Precision-Recall y ROC comparativas solo para TEST.

    La predicción binaria final del estudio se mantiene con umbral fijo 0.5.
    Estas curvas se calculan a partir de y_prob para visualizar el comportamiento
    probabilístico de cada estrategia a lo largo de distintos umbrales.

    Diseño de la figura:
    - Curva Precision-Recall arriba.
    - Curva ROC abajo.
    - Ambos ejes se representan con la misma escala 0-1.
    - Cada panel se fuerza a formato cuadrado para evitar interpretaciones visuales erróneas.
    - Leyendas en zonas propias, fuera de los ejes, para evitar solapes.
    - La leyenda muestra solo el código de colores y el nombre de cada estrategia.
    """
    if str(split_name).lower() != "test":
        return

    required_cols = {"scenario", "model_name", "y_true", "y_prob"}
    if predictions_df is None or predictions_df.empty or not required_cols.issubset(set(predictions_df.columns)):
        print("Curvas PR/ROC TEST omitidas: faltan columnas necesarias en predictions_df.")
        return

    predictions_df = predictions_df.copy()
    if "split" in predictions_df.columns:
        predictions_df = predictions_df[predictions_df["split"].astype(str).str.lower() == "test"].copy()

    if predictions_df.empty:
        print("Curvas PR/ROC TEST omitidas: no hay predicciones TEST.")
        return

    curves_dir = ensure_directory(FIGURES_DIR / "curvas_pr_roc_test")
    scenario_order = [s for s in get_scenarios().keys() if s in set(predictions_df["scenario"].unique())]
    model_order = [m for m in ALL_MODEL_NAMES if m in set(predictions_df["model_name"].unique())]

    for scenario in scenario_order:
        scenario_df = predictions_df[predictions_df["scenario"] == scenario].copy()
        if scenario_df.empty:
            continue

        y_true_all = pd.to_numeric(scenario_df["y_true"], errors="coerce").dropna().to_numpy(dtype=int)
        if len(np.unique(y_true_all)) < 2:
            print(f"Curvas PR/ROC TEST omitidas para {scenario}: solo hay una clase real.")
            continue

        positive_rate = float(np.mean(y_true_all == 1))

        fig = plt.figure(figsize=(9.2, 15.6))
        grid = fig.add_gridspec(
            nrows=4,
            ncols=1,
            height_ratios=[6.0, 1.05, 6.0, 1.05],
            hspace=0.32,
        )
        ax_pr = fig.add_subplot(grid[0, 0])
        ax_pr_legend = fig.add_subplot(grid[1, 0])
        ax_roc = fig.add_subplot(grid[2, 0])
        ax_roc_legend = fig.add_subplot(grid[3, 0])
        ax_pr_legend.axis("off")
        ax_roc_legend.axis("off")

        any_curve = False

        for model_name in model_order:
            model_df = scenario_df[scenario_df["model_name"] == model_name].copy()
            if model_df.empty:
                continue

            y_true = pd.to_numeric(model_df["y_true"], errors="coerce")
            y_prob = pd.to_numeric(model_df["y_prob"], errors="coerce")
            valid_mask = y_true.notna() & y_prob.notna()
            y_true = y_true[valid_mask].to_numpy(dtype=int)
            y_prob = y_prob[valid_mask].to_numpy(dtype=float)

            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                continue

            precision_values, recall_values, _ = precision_recall_curve(y_true, y_prob)
            fpr_values, tpr_values, _ = roc_curve(y_true, y_prob)

            color = MODEL_COLORS.get(model_name, "#777777")
            model_label = MODEL_LABELS.get(model_name, model_name)

            ax_pr.plot(
                recall_values,
                precision_values,
                linewidth=2.0,
                color=color,
                label=model_label,
            )
            ax_roc.plot(
                fpr_values,
                tpr_values,
                linewidth=2.0,
                color=color,
                label=model_label,
            )
            any_curve = True

        if not any_curve:
            plt.close(fig)
            continue

        ax_pr.axhline(
            positive_rate,
            linestyle="--",
            linewidth=1.2,
            color="#555555",
            alpha=0.80,
            label=f"Línea base positiva = {positive_rate:.3f}",
        )
        ax_roc.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.2,
            color="#555555",
            alpha=0.80,
            label="Clasificador aleatorio",
        )

        for ax in [ax_pr, ax_roc]:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal", adjustable="box")
            try:
                ax.set_box_aspect(1)
            except AttributeError:
                pass
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
            ax.set_yticks(np.linspace(0.0, 1.0, 6))
            ax.grid(linestyle="--", alpha=0.25)
            ax.set_axisbelow(True)

        ax_pr.set_xlabel("Recall / Sensibilidad")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Curva Precision-Recall", fontsize=13)

        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate / Recall")
        ax_roc.set_title("Curva ROC", fontsize=13)

        pr_handles, pr_labels = ax_pr.get_legend_handles_labels()
        roc_handles, roc_labels = ax_roc.get_legend_handles_labels()

        ax_pr_legend.legend(
            pr_handles,
            pr_labels,
            loc="center",
            ncol=3,
            fontsize=9.6,
            frameon=True,
            handlelength=2.4,
            columnspacing=1.2,
        )
        ax_roc_legend.legend(
            roc_handles,
            roc_labels,
            loc="center",
            ncol=3,
            fontsize=9.6,
            frameon=True,
            handlelength=2.4,
            columnspacing=1.2,
        )

        fig.suptitle(
            f"TEST · Curvas PR y ROC comparativas · {format_scenario_label_for_plot(scenario)}",
            fontsize=15,
            fontweight="bold",
            y=0.975,
        )

        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.035, top=0.940, hspace=0.32)
        fig.savefig(curves_dir / f"test_pr_roc_curves_{get_short_scenario_name(scenario)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

# ============================================================
# 8.1. SALIDAS RESUMIDAS: TABLAS LEGIBLES, TIEMPOS Y REMUESTREO
# ============================================================

def parse_duration_to_seconds(value, unit):
    value = float(value)
    unit = str(unit).strip().lower()
    if unit == "s":
        return value
    if unit == "min":
        return value * 60.0
    if unit == "h":
        return value * 3600.0
    return np.nan


def save_visual_table(table_df, output_path, title, fig_width=15.5):
    """Guarda una tabla como figura PNG para que sea directamente utilizable en la memoria."""
    if table_df is None or table_df.empty:
        table_df = pd.DataFrame({"Mensaje": ["Sin datos disponibles"]})

    table_df = table_df.copy().astype(str)
    fig_height = max(2.8, 0.42 * (len(table_df) + 2))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.28)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        elif row % 2 == 0:
            cell.set_facecolor("#fafafa")

    try:
        table.auto_set_column_width(col=list(range(len(table_df.columns))))
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def save_visual_table_by_scenario(table_df, output_path, title):
    """
    Guarda la tabla principal de tiempos en una figura amplia por escenario.

    En lugar de repetir la columna Escenario en todas las filas, se crea un
    panel por escenario. La figura se hace más grande y la tabla se escala para
    que sea legible directamente en la memoria o presentación.
    """
    if table_df is None or table_df.empty:
        save_visual_table(
            pd.DataFrame({"Mensaje": ["Sin datos disponibles"]}),
            output_path,
            title,
            fig_width=16.5,
        )
        return

    table_df = table_df.copy().astype(str)

    if "Escenario" not in table_df.columns:
        save_visual_table(table_df, output_path, title, fig_width=18.5)
        return

    scenario_order = [
        str(scenario_name)
        for scenario_name in get_scenarios().keys()
        if str(scenario_name) in set(table_df["Escenario"].astype(str))
    ]
    scenario_order += [
        s
        for s in table_df["Escenario"].astype(str).unique().tolist()
        if s not in scenario_order
    ]

    n_scenarios = max(1, len(scenario_order))
    ncols = 2 if n_scenarios > 1 else 1
    nrows = int(np.ceil(n_scenarios / ncols))

    # Mantiene la misma figura conjunta por escenarios, pero la hace más grande
    # para que cada tabla ocupe más espacio visual y se lea mejor.
    fig_width = 30.0 if ncols == 2 else 18.0
    fig_height = max(10.5, 8.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = np.asarray(axes).reshape(-1)

    fig.suptitle(title, fontsize=21, fontweight="bold", y=0.992)

    for ax_idx, ax in enumerate(axes):
        ax.axis("off")

        if ax_idx >= len(scenario_order):
            continue

        scenario_name = scenario_order[ax_idx]
        scenario_df = table_df[table_df["Escenario"].astype(str) == scenario_name].copy()
        scenario_df = scenario_df.drop(columns=["Escenario"], errors="ignore")

        if scenario_df.empty:
            ax.set_title(f"Escenario: {format_scenario_label_for_plot(scenario_name)}", fontsize=13.5, fontweight="bold", pad=10)
            continue

        ax.set_title(f"Escenario: {format_scenario_label_for_plot(scenario_name).upper()}", fontsize=16.5, fontweight="bold", pad=12)

        table = ax.table(
            cellText=scenario_df.values,
            colLabels=scenario_df.columns,
            cellLoc="center",
            bbox=[0.02, 0.08, 0.96, 0.72],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11.3)
        table.scale(1.55, 2.30)

        n_cols = len(scenario_df.columns)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#444444")
            cell.set_linewidth(0.46)

            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eeeeee")
            else:
                # Primeras tres filas: estrategias base. Resto: estrategias nuevas.
                # Se usa sombreado suave, pero sin negrita en ninguna estrategia.
                if row <= 3:
                    cell.set_facecolor("#ffffff" if row % 2 else "#f8f8f8")
                else:
                    cell.set_facecolor("#f3f3f3" if row % 2 else "#fbfbfb")

                # Línea algo más marcada al comenzar las estrategias nuevas,
                # sin aplicar negrita a Random Oversampling.
                if row == 4:
                    cell.set_linewidth(0.75)

                cell.set_text_props(weight="normal")

        try:
            table.auto_set_column_width(col=list(range(n_cols)))
        except Exception:
            pass

    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.955])
    fig.subplots_adjust(wspace=0.10, hspace=0.18)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

def print_comparison_table_by_scenario(df, split_name, decimals=4):
    """
    Imprime la tabla comparativa por escenario y por bloques de métricas.
    Así se evita una tabla gigante que se corta o se desordena en terminal.
    """
    if df is None or df.empty:
        print("\nNo hay tabla comparativa para imprimir.")
        return

    df_print = df.copy().round(decimals)
    blocks = [
        {
            "title": "Pesos / configuración",
            "cols": ["scenario", "model_name", "c0_mean", "c0_std", "c1_mean", "c1_std"],
        },
        {
            "title": "Métricas principales",
            "cols": [
                "scenario", "model_name",
                f"{split_name}_f1_mean", f"{split_name}_f1_std",
                f"{split_name}_pr_auc_mean", f"{split_name}_pr_auc_std",
                f"{split_name}_balanced_accuracy_mean", f"{split_name}_balanced_accuracy_std",
                f"{split_name}_mcc_mean", f"{split_name}_mcc_std",
            ],
        },
        {
            "title": "Métricas por clase y curvas",
            "cols": [
                "scenario", "model_name",
                f"{split_name}_recall_mean", f"{split_name}_recall_std",
                f"{split_name}_fnr_mean", f"{split_name}_fnr_std",
                f"{split_name}_specificity_mean", f"{split_name}_specificity_std",
                f"{split_name}_roc_auc_mean", f"{split_name}_roc_auc_std",
            ],
        },
    ]

    print("\n" + "=" * 110)
    print(f"TABLA COMPARATIVA FINAL · {split_name.upper()}")
    print("=" * 110)

    for scenario_name in get_scenarios().keys():
        temp_scenario = df_print[df_print["scenario"] == scenario_name].copy()
        if temp_scenario.empty:
            continue

        print("\n" + "#" * 110)
        print(f"ESCENARIO: {str(scenario_name).upper()}")
        print("#" * 110)

        for block in blocks:
            existing_cols = [col for col in block["cols"] if col in temp_scenario.columns]
            if len(existing_cols) <= 2:
                continue
            print("\n" + "-" * 110)
            print(block["title"])
            print("-" * 110)
            with pd.option_context("display.max_columns", None, "display.width", 240):
                print(temp_scenario[existing_cols].to_string(index=False))


def compute_resampling_effect_summary():
    """
    Calcula un resumen del efecto de cada estrategia de remuestreo sobre TRAIN.
    No guarda CSV y no entrena modelos: solo resume las cuentas antes/después.
    """
    rows = []

    for scenario_name, config in get_scenarios().items():
        for strategy_name in RESAMPLING_MODEL_NAMES:
            for seed in SEEDS:
                df = load_or_create_dataset(seed, scenario_name, config)
                target_col = get_target_col()
                train_df = df[df["split"] == "train"].reset_index(drop=True)
                y_train = train_df[target_col].to_numpy(dtype=int)

                before_counts = pd.Series(y_train).value_counts().to_dict()
                class0_before = int(before_counts.get(0, 0))
                class1_before = int(before_counts.get(1, 0))
                n_train_original = int(len(y_train))

                min_count = min(class0_before, class1_before)
                max_count = max(class0_before, class1_before)

                if strategy_name in ["random_oversampling", "smote"]:
                    class0_after = max_count
                    class1_after = max_count
                    n_train_resampled = int(class0_after + class1_after)
                    affected = int(n_train_resampled - n_train_original)
                    if strategy_name == "random_oversampling":
                        effect_type = "muestras duplicadas añadidas"
                    else:
                        effect_type = "muestras sintéticas añadidas"
                    k_neighbors = min(5, min_count - 1) if min_count > 1 and strategy_name == "smote" else np.nan
                elif strategy_name == "random_undersampling":
                    class0_after = min_count
                    class1_after = min_count
                    n_train_resampled = int(class0_after + class1_after)
                    affected = int(n_train_original - n_train_resampled)
                    effect_type = "muestras mayoritarias eliminadas"
                    k_neighbors = np.nan
                else:
                    class0_after = np.nan
                    class1_after = np.nan
                    n_train_resampled = np.nan
                    affected = np.nan
                    effect_type = "no definido"
                    k_neighbors = np.nan

                rows.append({
                    "scenario": scenario_name,
                    "strategy": strategy_name,
                    "seed": seed,
                    "n_train_original": n_train_original,
                    "class0_before": class0_before,
                    "class1_before": class1_before,
                    "n_train_resampled": n_train_resampled,
                    "class0_after": class0_after,
                    "class1_after": class1_after,
                    "affected_samples": affected,
                    "effect_type": effect_type,
                    "k_neighbors": k_neighbors,
                })

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, raw_df

    agg_rows = []
    for (scenario_name, strategy_name), group in raw_df.groupby(["scenario", "strategy"]):
        effect_type = group["effect_type"].iloc[0]
        agg_rows.append({
            "escenario": scenario_name,
            "estrategia": strategy_name,
            "n_train_original_medio": float(group["n_train_original"].mean()),
            "clase0_antes_media": float(group["class0_before"].mean()),
            "clase1_antes_media": float(group["class1_before"].mean()),
            "n_train_remuestreado_medio": float(group["n_train_resampled"].mean()),
            "clase0_despues_media": float(group["class0_after"].mean()),
            "clase1_despues_media": float(group["class1_after"].mean()),
            "muestras_afectadas_media": float(group["affected_samples"].mean()),
            "tipo_efecto": effect_type,
            "k_neighbors_smote_medio": float(group["k_neighbors"].dropna().mean()) if group["k_neighbors"].notna().any() else np.nan,
        })

    summary_df = pd.DataFrame(agg_rows).sort_values(["escenario", "estrategia"]).reset_index(drop=True)
    return raw_df, summary_df


def save_and_format_resampling_summary():
    """Guarda y devuelve el resumen de remuestreo solo en TXT, sin CSV ni PNG."""
    ensure_directory(RESAMPLING_INFO_DIR)
    _, summary_df = compute_resampling_effect_summary()
    summary_text_path = RESAMPLING_INFO_DIR / "resumen_remuestreo.txt"
    if summary_df.empty:
        text = "RESUMEN DE REMUESTREO\n" + "=" * 110 + "\nSin datos de remuestreo."
        with open(summary_text_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text

    view = summary_df.copy()
    numeric_cols = [
        "n_train_original_medio", "clase0_antes_media", "clase1_antes_media",
        "n_train_remuestreado_medio", "clase0_despues_media", "clase1_despues_media",
        "muestras_afectadas_media", "k_neighbors_smote_medio",
    ]
    for col in numeric_cols:
        if col in view.columns:
            view[col] = view[col].round(2)


    lines = []
    lines.append("RESUMEN DE REMUESTREO · ESTRATEGIAS NUEVAS")
    lines.append("=" * 110)
    lines.append("Lectura rápida:")
    lines.append("- Random Oversampling: iguala la clase minoritaria duplicando muestras del TRAIN.")
    lines.append("- SMOTE: iguala la clase minoritaria generando muestras sintéticas/interpoladas en TRAIN.")
    lines.append("- Random Undersampling: reduce la clase mayoritaria eliminando muestras del TRAIN.")
    lines.append("- VALIDACIÓN y TEST no se remuestrean en ningún caso.")
    lines.append("")

    for scenario_name in get_scenarios().keys():
        temp = view[view["escenario"] == scenario_name].copy()
        if temp.empty:
            continue
        lines.append("#" * 110)
        lines.append(f"ESCENARIO: {str(scenario_name).upper()}")
        lines.append("#" * 110)
        with pd.option_context("display.max_columns", None, "display.width", 220):
            lines.append(temp.to_string(index=False))
        lines.append("")

    lines.append("Archivo de remuestreo guardado en:")
    lines.append(f"- {summary_text_path}")

    text = "\n".join(lines)
    with open(summary_text_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text



def parse_base_timer_totals():
    """
    Recupera los tiempos reales de las estrategias base desde el archivo
    timers_execution.txt generado por el código WBCE correspondiente.

    No mide tiempos de carga en este script. Extrae el coste que ya se calculó
    en el flujo base y lo adapta a la tabla final:
    TRAIN real, validación rejilla, agregación/selección Best F1, TEST final y total comparable.
    """
    timers_path = BASE_OUTPUT_DIR / "timers_execution.txt"
    rows = []

    if not timers_path.exists():
        return pd.DataFrame(rows)

    try:
        text = timers_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame(rows)

    section = text
    marker_start = "0.0) COSTE COMPUTACIONAL COMPARABLE HASTA TEST"
    marker_end = "0.1)"
    if marker_start in text:
        section = text.split(marker_start, 1)[1]
        if marker_end in section:
            section = section.split(marker_end, 1)[0]

    aliases = {
        "best_val_f1_grid": ["Weighted grid / Best F1", "Weighted Best F1", "Best F1"],
        "equal_weights": ["Equal weights"],
        "sklearn_balanced": ["Sklearn balanced"],
    }

    duration_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(s|min|h)")

    for scenario_name in get_scenarios().keys():
        for model_name, alias_list in aliases.items():
            matching_lines = []
            for line in section.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if str(scenario_name) not in stripped:
                    continue
                if any(alias in stripped for alias in alias_list):
                    matching_lines.append(stripped)

            if not matching_lines:
                for line in text.splitlines():
                    stripped = line.strip()
                    if str(scenario_name) in stripped and any(alias in stripped for alias in alias_list):
                        matching_lines.append(stripped)

            base_row = {
                "escenario": scenario_name,
                "estrategia": model_name,
                "fuente": "código base WBCE",
                "estado": "tiempo base recuperado",
                "modelos_entrenados": np.nan,
                "entrenamiento_segundos": np.nan,
                "validacion_segundos": np.nan,
                "agregacion_seleccion_segundos": np.nan,
                "test_segundos": np.nan,
                "total_segundos": np.nan,
                "validacion_texto": "No aplica",
                "agregacion_seleccion_texto": "No aplica",
            }

            if not matching_lines:
                base_row["estado"] = "no recuperado"
                rows.append(base_row)
                continue

            line = matching_lines[0]
            durations = [
                parse_duration_to_seconds(m.group(1), m.group(2))
                for m in duration_pattern.finditer(line)
            ]

            if model_name == "best_val_f1_grid" and len(durations) >= 5:
                # Tabla comparable del código base:
                # TRAIN rejilla | validación rejilla | agregación/selección Best F1 | TEST final | total comparable.
                base_row.update({
                    "entrenamiento_segundos": durations[-5],
                    "validacion_segundos": durations[-4],
                    "agregacion_seleccion_segundos": durations[-3],
                    "test_segundos": durations[-2],
                    "total_segundos": durations[-1],
                    "validacion_texto": format_seconds(durations[-4]),
                    "agregacion_seleccion_texto": format_seconds(durations[-3]),
                })

            elif model_name in ["equal_weights", "sklearn_balanced"] and len(durations) >= 3:
                # Tabla comparable del código base:
                # TRAIN real | TEST final | total comparable.
                base_row.update({
                    "entrenamiento_segundos": durations[-3],
                    "test_segundos": durations[-2],
                    "total_segundos": durations[-1],
                })
            else:
                base_row["estado"] = "no recuperado"

            rows.append(base_row)

    return pd.DataFrame(rows)


def save_timer_summary(timer_frames, total_start):
    """
    Muestra por terminal un resumen legible de tiempos y guarda exactamente ese
    resumen en TXT. También crea una figura PNG compacta con el coste por
    estrategia.

    La tabla final usa únicamente tiempos reales/wall-clock por bloques útiles:
    TRAIN real, validación rejilla, agregación/selección Best F1, TEST final y total comparable.
    No se muestran ni se suman tiempos acumulados internos de las seeds.
    """
    script_total_seconds = time.perf_counter() - total_start

    timer_frames.append(pd.DataFrame([{
        "scenario": "all",
        "seed": "all",
        "stage": "script_total",
        "model_name": "wall_clock",
        "n_models": np.nan,
        "total_seconds": script_total_seconds,
        "total_time_readable": format_seconds(script_total_seconds),
        "skipped_because_existing_outputs": False,
    }]))

    timers_df = pd.concat(
        [df for df in timer_frames if df is not None and not df.empty],
        ignore_index=True,
    )

    ensure_directory(TIMES_DIR)
    summary_text_path = TIMES_DIR / "resumen_tiempos_computacion.txt"
    summary_figure_path = TIMES_DIR / "tabla_tiempos_computacion.png"
    main_time_table_path = TIMES_DIR / "tabla_tiempos_computacion.csv"

    wall_rows = timers_df[timers_df["seed"].astype(str) == "all"].copy()

    def get_wall_seconds(stage_name, scenario_name=None, model_name=None):
        rows = wall_rows[wall_rows["stage"].astype(str) == str(stage_name)].copy()
        if scenario_name is not None:
            rows = rows[rows["scenario"].astype(str) == str(scenario_name)]
        if model_name is not None:
            rows = rows[rows["model_name"].astype(str) == str(model_name)]
        if rows.empty or "total_seconds" not in rows.columns:
            return np.nan
        values = pd.to_numeric(rows["total_seconds"], errors="coerce")
        if values.notna().sum() == 0:
            return np.nan
        return float(values.sum())

    def get_wall_n_models(stage_name, scenario_name=None, model_name=None, fallback=None):
        rows = wall_rows[wall_rows["stage"].astype(str) == str(stage_name)].copy()
        if scenario_name is not None:
            rows = rows[rows["scenario"].astype(str) == str(scenario_name)]
        if model_name is not None:
            rows = rows[rows["model_name"].astype(str) == str(model_name)]

        if rows.empty or "n_models" not in rows.columns:
            if fallback is not None and not pd.isna(fallback):
                return int(fallback)
            return 0

        value = pd.to_numeric(rows["n_models"], errors="coerce").fillna(0.0).sum()

        if value == 0 and fallback is not None and not pd.isna(fallback):
            return int(fallback)

        return int(value)

    def format_or_dash(value):
        if pd.isna(value):
            return "-"
        return format_seconds(value)

    def format_no_aplica_or_time(value, text_value=""):
        if str(text_value).strip().lower() == "no aplica":
            return "No aplica"
        return format_or_dash(value)

    def safe_sum_required(values):
        if any(pd.isna(v) for v in values):
            return np.nan
        return float(sum(float(v) for v in values))

    # 1) Tiempos reales de las estrategias base recuperados del código WBCE.
    base_time_df = parse_base_timer_totals()
    if not base_time_df.empty:
        base_time_df["train_real_segundos"] = base_time_df["entrenamiento_segundos"]
        base_time_df["modelos_entrenados"] = [
            get_expected_train_models_for_strategy(strategy_name, scenario_name)
            for strategy_name, scenario_name in zip(base_time_df["estrategia"], base_time_df["escenario"])
        ]

    # 2) Tiempos de estrategias nuevas calculados en este script.
    new_rows = []
    for scenario_name in get_scenarios().keys():
        for strategy_name in TRAINED_COMPARISON_MODEL_NAMES:
            train_stage = "greedy_class_weight_train_parallel_total" if strategy_name == "greedy_class_weight" else f"{strategy_name}_train_parallel_total"
            test_stage = f"{strategy_name}_test_parallel_total"

            train_real = get_wall_seconds(train_stage, scenario_name, strategy_name)
            test_seconds = get_wall_seconds(test_stage, scenario_name, strategy_name)

            coefficients_reused_stage = "load_existing_greedy_class_weight_coefficients" if strategy_name == "greedy_class_weight" else "load_existing_resampling_coefficients"
            coefficients_reused = not pd.isna(get_wall_seconds(
                coefficients_reused_stage, scenario_name, strategy_name
            )) and pd.isna(train_real)

            if coefficients_reused:
                estado = "coeficientes reutilizados; tiempo TRAIN no medido en esta ejecución"
                total_comparable = np.nan
            else:
                estado = "entrenado"
                total_comparable = safe_sum_required([train_real, test_seconds])

            new_rows.append({
                "escenario": scenario_name,
                "estrategia": strategy_name,
                "fuente": "script comparación",
                "estado": estado,
                "modelos_entrenados": get_wall_n_models(
                    train_stage,
                    scenario_name,
                    strategy_name,
                    fallback=get_expected_train_models_for_strategy(strategy_name, scenario_name),
                ),
                "train_real_segundos": train_real,
                "validacion_segundos": np.nan,
                "agregacion_seleccion_segundos": np.nan,
                "test_segundos": test_seconds,
                "total_segundos": total_comparable,
                "validacion_texto": "No aplica",
                "agregacion_seleccion_texto": "No aplica",
            })

    new_time_df = pd.DataFrame(new_rows)
    global_time_df = pd.concat([base_time_df, new_time_df], ignore_index=True) if not base_time_df.empty else new_time_df.copy()

    if not global_time_df.empty:
        global_time_df["estrategia_label"] = global_time_df["estrategia"].map(MODEL_LABELS).fillna(global_time_df["estrategia"])
        global_time_df["train_real"] = global_time_df["train_real_segundos"].apply(format_or_dash)
        global_time_df["validacion"] = [
            format_no_aplica_or_time(v, txt)
            for v, txt in zip(
                global_time_df["validacion_segundos"],
                global_time_df.get("validacion_texto", pd.Series([""] * len(global_time_df))),
            )
        ]
        global_time_df["agregacion/seleccion"] = [
            format_no_aplica_or_time(v, txt)
            for v, txt in zip(
                global_time_df["agregacion_seleccion_segundos"],
                global_time_df.get("agregacion_seleccion_texto", pd.Series([""] * len(global_time_df))),
            )
        ]
        global_time_df["test_final"] = global_time_df["test_segundos"].apply(format_or_dash)
        global_time_df["total_comparable"] = global_time_df["total_segundos"].apply(format_or_dash)
    main_visual_table = global_time_df[[
        "escenario",
        "estrategia_label",
        "modelos_entrenados",
        "train_real",
        "validacion",
        "agregacion/seleccion",
        "test_final",
        "total_comparable",
    ]].copy() if not global_time_df.empty else pd.DataFrame()

    if not main_visual_table.empty:
        main_visual_table.columns = [
            "Escenario",
            "Estrategia",
            "Modelos entrenados TRAIN",
            "TRAIN real",
            "Validación rejilla",
            "Agregación/selección Best F1",
            "TEST final",
            "Total comparable",
        ]
        main_visual_table.to_csv(main_time_table_path, index=False)

        save_visual_table_by_scenario(
            main_visual_table,
            summary_figure_path,
            "Coste computacional comparable por estrategia · tiempos reales por bloques",
        )
    else:
        save_visual_table(
            pd.DataFrame({"Mensaje": ["Sin datos de tiempos"]}),
            summary_figure_path,
            "Coste computacional comparable",
        )

    # Resumen de remuestreo solo en TXT.
    resampling_text = save_and_format_resampling_summary()

    lines = []
    lines.append("RESUMEN DE TIEMPOS · COMPARACIÓN FINAL DE ESTRATEGIAS")
    lines.append("=" * 120)
    lines.append("Lectura rápida:")
    lines.append(f"- Nuevas estrategias incluidas: {', '.join(TRAINED_COMPARISON_MODEL_NAMES)}.")
    lines.append("- Todas usan las mismas 20 seeds/particiones que el código base.")
    lines.append("- La heurística greedy busca pesos solo dentro de TRAIN mediante CV interna de 5 folds; VALIDACIÓN y TEST no intervienen en el ajuste.")
    lines.append("- El remuestreo se aplica solo al TRAIN; VALIDACIÓN y TEST no se remuestrean.")
    lines.append("- Para WBCE Best F1, Equal weights y Sklearn balanced se recuperan sus tiempos reales desde timers_execution.txt del código base.")
    lines.append("- En la evaluación final no se reentrena: las estrategias base cargan resultados/predicciones del flujo WBCE y las nuevas cargan sus coeficientes guardados.")
    lines.append("- Antes de medir se realiza un warm-up técnico realista: precarga datasets e inicializa pandas, NumPy/BLAS, métricas, joblib, imblearn y sklearn para que el primer bloque medido no quede inflado.")
    lines.append("- En WBCE Heurística, TRAIN real incluye preparar TRAIN, escalar, búsqueda greedy con CV interna de 5 folds, refit en TRAIN y guardar coeficientes.")
    lines.append("- En las estrategias de remuestreo, TRAIN real incluye preparar TRAIN, escalar, remuestrear solo TRAIN, entrenar LogisticRegression y guardar coeficientes.")
    lines.append("- No se muestran tiempos acumulados internos de las 20 seeds ni se separa artificialmente remuestreo/entrenamiento.")
    lines.append("- La evaluación en VAL de las estrategias nuevas se calcula para tablas/figuras de validación, pero no se incluye en el coste comparable porque no hay selección de hiperparámetros.")
    lines.append("- Si ya existen los CSV de comparación de VAL/TEST, se cargan directamente y no se recalculan esas evaluaciones.")
    lines.append("")

    lines.append("0.0) COSTE COMPUTACIONAL COMPARABLE GLOBAL · BASE + NUEVAS ESTRATEGIAS")
    lines.append("   Columnas: TRAIN real + validación rejilla + agregación/selección Best F1 + TEST final + total comparable.")
    lines.append("   Para WBCE Best F1 se incluye TRAIN de la rejilla, validación de la rejilla, selección del mejor peso, salidas guardadas y TEST final.")
    lines.append("   Para Equal weights y Sklearn balanced se incluye TRAIN real del código base, sus salidas guardadas y TEST final; no tienen búsqueda de rejilla.")
    lines.append("   Para WBCE Heurística, TRAIN real incluye preparar TRAIN, escalar, búsqueda greedy con CV interna de 5 folds, refit en TRAIN y guardar coeficientes; después se evalúa en TEST.")
    lines.append("   Para Random Oversampling, SMOTE y Random Undersampling, TRAIN real incluye preparar TRAIN, escalar, remuestrear solo TRAIN, entrenar y guardar coeficientes; después se evalúa en TEST.")

    if global_time_df.empty:
        lines.append("   - Sin datos.")
    else:
        for scenario_name in get_scenarios().keys():
            temp = global_time_df[global_time_df["escenario"] == scenario_name][[
                "escenario",
                "estrategia_label",
                "modelos_entrenados",
                "train_real",
                "validacion",
                "agregacion/seleccion",
                "test_final",
                "total_comparable",
            ]].copy()
            if temp.empty:
                continue
            lines.append("")
            lines.append("#" * 120)
            lines.append(f"ESCENARIO: {str(scenario_name).upper()}")
            lines.append("#" * 120)
            temp.columns = [
                "escenario",
                "estrategia",
                "modelos entrenados TRAIN",
                "TRAIN real",
                "validación rejilla",
                "agregación/selección Best F1",
                "test final",
                "total comparable",
            ]
            with pd.option_context("display.max_columns", None, "display.width", 220):
                lines.append(temp.to_string(index=False))

    lines.append("")
    lines.append("0.1) SCRIPT COMPLETO")
    lines.append(f"   Tiempo total del script de comparación: {format_or_dash(get_wall_seconds('script_total', 'all', 'wall_clock'))}")
    lines.append("")
    lines.append("Archivos de tiempos guardados en:")
    lines.append(f"- {summary_text_path}")
    lines.append(f"- {summary_figure_path}")
    lines.append(f"- {main_time_table_path}")
    lines.append("- La tabla principal de tiempos se guarda en CSV/PNG; el resumen de remuestreo queda al final de este TXT.")

    lines.append("")
    lines.append(resampling_text)

    summary_text = "\n".join(lines)
    with open(summary_text_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n\n" + summary_text)
    return timers_df


def get_comparison_output_paths(split_name):
    """Rutas de resultados ya guardados por este script de comparación."""
    return {
        "raw": OUTPUT_DIR / f"{split_name}_all_results_raw_multiseed.csv",
        "predictions": OUTPUT_DIR / f"{split_name}_all_predictions_raw_multiseed.csv",
        "aggregated": OUTPUT_DIR / f"{split_name}_all_results_aggregated_mean_std.csv",
        "comparison": OUTPUT_DIR / f"{split_name}_all_comparison_table.csv",
    }


def comparison_split_outputs_are_compatible(split_name):
    """
    Comprueba si ya existen los resultados completos de VAL/TEST de la comparación.

    Si existen, el script los carga directamente y NO vuelve a evaluar ese split.
    Esto replica la lógica de los códigos WBCE base: si ya están guardadas las
    salidas finales, se reutilizan para evitar repetir cálculos innecesarios.
    """
    paths = get_comparison_output_paths(split_name)

    metadata_path = OUTPUT_DIR / "comparison_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return False
    if metadata.get("comparison_run_signature") != COMPARISON_RUN_SIGNATURE:
        return False

    if not all(path.exists() for path in paths.values()):
        return False

    try:
        raw_index = pd.read_csv(paths["raw"], usecols=["scenario", "model_name"])
        pred_head = pd.read_csv(paths["predictions"], nrows=5)
        agg_head = pd.read_csv(paths["aggregated"], nrows=5)
        comp_head = pd.read_csv(paths["comparison"], nrows=5)
    except Exception:
        return False

    expected_scenarios = set(get_scenarios().keys())
    existing_scenarios = set(raw_index["scenario"].astype(str).unique().tolist())
    if existing_scenarios != set(str(s) for s in expected_scenarios):
        return False

    expected_models = set(ALL_MODEL_NAMES)
    existing_models = set(raw_index["model_name"].astype(str).unique().tolist())
    if existing_models != expected_models:
        return False

    required_pred_cols = {"scenario", "seed", "model_name", "sample_id", "y_true", "y_pred", "y_prob"}
    if not required_pred_cols.issubset(set(pred_head.columns)):
        return False

    required_agg_cols = {"scenario", "model_name", f"{split_name}_f1_mean"}
    if not required_agg_cols.issubset(set(agg_head.columns)):
        return False

    required_comp_cols = {"scenario", "model_name", f"{split_name}_f1_mean"}
    if not required_comp_cols.issubset(set(comp_head.columns)):
        return False

    return True


def load_or_build_comparison_outputs_for_split(split_name, scenarios, timer_frames):
    """
    Carga o genera los resultados completos de comparación para VAL o TEST.

    - Si ya existen los CSV de comparación, se cargan directamente.
    - Si faltan, se cargan los resultados base WBCE y solo se evalúan las
      estrategias nuevas para completar la tabla.
    """
    paths = get_comparison_output_paths(split_name)

    if comparison_split_outputs_are_compatible(split_name):
        t_load_start = time.perf_counter()
        print("\n" + "=" * 100)
        print(f"{split_name.upper()} · Resultados de comparación ya existentes")
        print("=" * 100)
        print("Se cargan directamente los CSV guardados y se salta la evaluación de este split.")
        print("No se reentrena y no se recalculan predicciones/métricas de VAL/TEST.")

        all_results = pd.read_csv(paths["raw"])
        all_predictions = pd.read_csv(paths["predictions"])
        agg_df = pd.read_csv(paths["aggregated"])
        comparison_df = pd.read_csv(paths["comparison"])

        timer_frames.append(pd.DataFrame([{
            "scenario": "all",
            "seed": "all",
            "stage": f"load_existing_comparison_{split_name}_outputs",
            "model_name": "wall_clock",
            "n_models": 0,
            "total_seconds": time.perf_counter() - t_load_start,
            "total_time_readable": format_seconds(time.perf_counter() - t_load_start),
            "skipped_because_existing_outputs": True,
        }]))

        return all_results, all_predictions, agg_df, comparison_df, paths

    print("\n" + "=" * 100)
    print(f"{split_name.upper()} · No existen resultados completos de comparación")
    print("=" * 100)
    print("Se evalúan las estrategias nuevas y se guardan los CSV para futuras ejecuciones.")

    base_results, base_predictions = load_base_results_and_predictions(split_name)
    new_results_frames = []
    new_predictions_frames = []

    for scenario_name, config in scenarios.items():
        for strategy_name in TRAINED_COMPARISON_MODEL_NAMES:
            res_df, pred_df = evaluate_resampling_strategy(scenario_name, config, strategy_name, split_name, timer_frames)
            new_results_frames.append(res_df)
            new_predictions_frames.append(pred_df)

    resampling_results = pd.concat(new_results_frames, ignore_index=True)
    resampling_predictions = pd.concat(new_predictions_frames, ignore_index=True)

    all_results = pd.concat([base_results, resampling_results], ignore_index=True)
    all_predictions = pd.concat([base_predictions, resampling_predictions], ignore_index=True)

    model_rank = {m: i for i, m in enumerate(ALL_MODEL_NAMES)}
    all_results["_model_rank"] = all_results["model_name"].map(model_rank).fillna(999)
    all_results = all_results.sort_values(["scenario", "_model_rank", "seed"]).drop(columns="_model_rank").reset_index(drop=True)
    all_predictions["_model_rank"] = all_predictions["model_name"].map(model_rank).fillna(999)
    all_predictions = all_predictions.sort_values(["scenario", "_model_rank", "seed", "sample_id"]).drop(columns="_model_rank").reset_index(drop=True)

    all_results.to_csv(paths["raw"], index=False)
    all_predictions.to_csv(paths["predictions"], index=False)

    agg_df = aggregate_results(all_results, ["scenario", "model_name"])
    comparison_df = make_comparison_table(agg_df, split_name)
    agg_df.to_csv(paths["aggregated"], index=False)
    comparison_df.to_csv(paths["comparison"], index=False)

    return all_results, all_predictions, agg_df, comparison_df, paths

def run_comparison():
    total_start = time.perf_counter()
    ensure_directory(OUTPUT_DIR)
    ensure_directory(FIGURES_DIR)
    ensure_directory(COEFFICIENTS_DIR)
    ensure_directory(RESAMPLING_INFO_DIR)
    ensure_directory(TIMES_DIR)

    print("\n" + "#" * 100)
    print("COMPARACIÓN FINAL DE ESTRATEGIAS")
    print("#" * 100)
    print(f"Salida: {OUTPUT_DIR.resolve()}")
    print("Estrategias nuevas: WBCE Heurística, RandomOverSampler, SMOTE y RandomUnderSampler.")
    print("Todas usan las mismas 20 seeds. La heurística greedy y el remuestreo se ajustan solo con TRAIN.\n")

    validate_required_base_flow()
    run_parallel_warmup()

    timer_frames = []
    scenarios = get_scenarios()

    # 1) Entrenar/cargar coeficientes de estrategias nuevas.
    for scenario_name, config in scenarios.items():
        train_or_load_greedy_strategy(scenario_name, config, timer_frames)
        for strategy_name in RESAMPLING_MODEL_NAMES:
            train_or_load_resampling_strategy(scenario_name, config, strategy_name, timer_frames)

    # 2) Cargar o generar resultados completos de VAL y TEST.
    #    Si los CSV ya existen, se cargan y se salta la evaluación, igual que en los códigos WBCE base.
    for split_name in ["val", "test"]:
        all_results, all_predictions, agg_df, comparison_df, _ = load_or_build_comparison_outputs_for_split(
            split_name=split_name,
            scenarios=scenarios,
            timer_frames=timer_frames,
        )

        print_comparison_table_by_scenario(comparison_df, split_name, decimals=4)

        save_bar_comparison_plots(agg_df, split_name)
        save_confusion_matrix_panels(all_predictions, split_name)
        save_boxplot_plots(all_results, split_name)
        save_pr_roc_curve_panels(all_predictions, all_results, split_name)


    metadata = {
        "description": "Comparación final de WBCE frente a heurística greedy de pesos y estrategias clásicas de remuestreo.",
        "comparison_run_signature": COMPARISON_RUN_SIGNATURE,
        "base_output_dir": str(BASE_OUTPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "seeds": SEEDS,
        "greedy_strategy": GREEDY_MODEL_NAMES,
        "resampling_strategies": RESAMPLING_MODEL_NAMES,
        "base_strategies_loaded": BASE_MODEL_NAMES,
        "model": "LogisticRegression L2 con max_iter=500; greedy usa CV interna 5 folds solo en TRAIN y remuestreo usa LogisticRegression sin class_weight",
        "greedy_policy": "GreedyClassWeightLogisticRegressionCV con weight_bounds=(0.01, 10), cv=StratifiedKFold(n_splits=5), scoring=make_scorer(f1_score, zero_division=0), n_jobs=1 y refit=True",
        "resampling_policy": "fit_resample solo sobre TRAIN; VALIDACIÓN y TEST no se remuestrean",
        "max_iter": MODEL_MAX_ITER,
    }
    with open(OUTPUT_DIR / "comparison_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    save_timer_summary(timer_frames, total_start)

    print("\nTodo terminado correctamente.")
    print("Resultados, coeficientes, tablas y figuras guardados en:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    run_comparison()

