
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from itertools import product
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
TRAIN_SIZE_FINAL = 0.72
VAL_SIZE_FINAL = 0.18
TEST_SIZE_FINAL = 0.10

# Rejilla completa: len(C_VALUES) x len(C_VALUES) combinaciones de pesos por semilla. (pesos c1 y c0)
C_VALUES = np.round(np.logspace(np.log10(0.01), np.log10(10), 30), 6).tolist() # 30 valores log-espaciados entre 0.01 y 10, redondeados a 6 decimales.
N_JOBS = -1
MODEL_MAX_ITER = 500

EQUAL_WEIGHTS_C0 = 1.0
EQUAL_WEIGHTS_C1 = 1.0

GRID_MODEL_NAME = "grid_manual"
FINAL_MODEL_NAMES = ["best_val_f1_grid", "equal_weights", "sklearn_balanced"]

MODEL_COLORS = {
    "best_val_f1_grid": "#D55E00",
    "sklearn_balanced": "#0072B2",
    "equal_weights": "#CC79A7",
}

PIN_EDGE_COLOR = "black"
PIN_TEXT_COLOR = "black"

MODEL_LABELS = {
    "best_val_f1_grid": "Manual (Best F1)",
    "sklearn_balanced": "Sklearn balanced",
    "equal_weights": "Equal weights",
}

DATA_FILE_NAME = "SuicideRisk_SecundarySchool_Spain_2025.xlsx"
EXCEL_SHEET_NAME = 0
ID_COL = "ID"

# Risk 3 y Risk 4 se eliminan del flujo experimental porque tienen muy pocos
# positivos para garantizar particiones clásicas de validación y test útiles.
TARGET_COLS = ["Suicide_Risk1", "Suicide_Risk2"]
REAL_SCENARIO_PREFIX = "real_2"

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE_PATH = PROJECT_DIR / DATA_FILE_NAME

OUTPUT_DIR = PROJECT_DIR / "outputs_dataset_real_2_particion_clasica_risk1_risk2"
FIGURES_DIR = OUTPUT_DIR / "figures"
GLOBAL_FIGURES_DIR = FIGURES_DIR / "comparativas_globales"
DATASETS_DIR = OUTPUT_DIR / "datasets"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
COEFFICIENTS_WIDE_DIR = OUTPUT_DIR / "coefficients_wide"
GRID_MANUAL_COEFFICIENTS_WIDE_DIR = COEFFICIENTS_WIDE_DIR / "grid_manual"
EQUAL_WEIGHTS_COEFFICIENTS_WIDE_DIR = COEFFICIENTS_WIDE_DIR / "equal_weights"
SKLEARN_BALANCED_COEFFICIENTS_WIDE_DIR = COEFFICIENTS_WIDE_DIR / "sklearn_balanced"


for folder in [
    OUTPUT_DIR, FIGURES_DIR, GLOBAL_FIGURES_DIR, DATASETS_DIR, ANALYSIS_DIR,
    COEFFICIENTS_WIDE_DIR, GRID_MANUAL_COEFFICIENTS_WIDE_DIR,
    EQUAL_WEIGHTS_COEFFICIENTS_WIDE_DIR, SKLEARN_BALANCED_COEFFICIENTS_WIDE_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

RUN_SIGNATURE = "real_2_risk1_risk2_classic_split_grid30_v4_mean_std_legend_cm_fixed"

SCENARIOS = {
    f"{REAL_SCENARIO_PREFIX}_{target_col.lower()}": {
        "dataset": DATA_FILE_NAME,
        "target": target_col,
        "type": "real_dataset_2_quantitative_classic_split",
        "description": f"Dataset real 2 con partición clásica para predecir {target_col}.",
    }
    for target_col in TARGET_COLS
}

PATHS = {
    "metadata": OUTPUT_DIR / "experiment_metadata.json",
    "timers": OUTPUT_DIR / "timers_execution.txt",
    "scenarios_config": OUTPUT_DIR / "scenarios_config.csv",
    "val_grid_raw": OUTPUT_DIR / "val_grid_results_raw_multiseed.csv",
    "val_grid_pred": OUTPUT_DIR / "val_grid_predictions_raw_multiseed.csv",
    "val_grid_agg": OUTPUT_DIR / "val_grid_results_aggregated_mean_std.csv",
    "best_configs": OUTPUT_DIR / "best_configs_by_val_f1_mean.csv",
    "balanced_raw": OUTPUT_DIR / "balanced_sklearn_points_raw_by_seed.csv",
    "balanced": OUTPUT_DIR / "balanced_sklearn_points_by_scenario.csv",
    "val_final_raw": OUTPUT_DIR / "val_final_results_raw_multiseed.csv",
    "val_final_pred": OUTPUT_DIR / "val_final_predictions_selected_models_raw_multiseed.csv",
    "val_final_agg": OUTPUT_DIR / "val_final_results_aggregated_mean_std.csv",
    "val_final_comp": OUTPUT_DIR / "val_final_comparison_table.csv",
    "test_final_raw": OUTPUT_DIR / "test_final_results_raw_multiseed.csv",
    "test_final_pred": OUTPUT_DIR / "test_final_predictions_raw_multiseed.csv",
    "test_final_agg": OUTPUT_DIR / "test_final_results_aggregated_mean_std.csv",
    "test_final_comp": OUTPUT_DIR / "test_final_comparison_table.csv",
}

iba_metric = make_index_balanced_accuracy(alpha=0.1, squared=True)(geometric_mean_score)


# ============================================================
# 1. UTILIDADES GENERALES
# ============================================================

def scenario_short(scenario_name):
    return str(scenario_name).replace("real_2_suicide_risk", "r2_risk")


def scenario_figures_dir(scenario_name):
    path = FIGURES_DIR / str(scenario_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} min"
    return f"{minutes / 60:.2f} h"


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


def sigmoid_stable(logits):
    logits = np.clip(np.asarray(logits, dtype=float), -709, 709)
    return 1.0 / (1.0 + np.exp(-logits))


def make_logistic_model(class_weight, seed):
    return LogisticRegression(
        random_state=int(seed),
        class_weight=class_weight,
        max_iter=MODEL_MAX_ITER,
    )


def fit_model_with_convergence_info(model, X_train_scaled, y_train):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train_scaled, y_train)

    messages = [
        str(w.message)
        for w in caught_warnings
        if issubclass(w.category, ConvergenceWarning)
    ]
    n_iter = int(np.max(model.n_iter_)) if hasattr(model, "n_iter_") else np.nan
    max_iter = int(model.max_iter) if hasattr(model, "max_iter") else np.nan
    reached_max_iter = bool(n_iter >= max_iter) if not pd.isna(n_iter) and not pd.isna(max_iter) else False

    return {
        "n_iter": n_iter,
        "max_iter": max_iter,
        "convergence_warning": len(messages) > 0,
        "reached_max_iter": reached_max_iter,
        "converged_without_warning": len(messages) == 0,
        "convergence_message": " | ".join(messages),
    }


def compute_sklearn_balanced_weights(y_train):
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(y_train, dtype=int))
    return float(weights[0]), float(weights[1])


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
        f"{split_name}_tn": int(tn),
        f"{split_name}_fp": int(fp),
        f"{split_name}_fn": int(fn),
        f"{split_name}_tp": int(tp),
    }

    return metrics


def add_wall_clock_timer(timer_frames, scenario_name, stage, start_time, n_models=None, skipped=False):
    elapsed = time.perf_counter() - start_time
    timer_frames.append(pd.DataFrame([{
        "scenario": scenario_name,
        "target": SCENARIOS.get(scenario_name, {}).get("target", "all"),
        "seed": "all",
        "stage": stage,
        "model_name": "wall_clock",
        "n_models": n_models if n_models is not None else np.nan,
        "total_seconds": elapsed,
        "total_time_readable": format_seconds(elapsed),
        "skipped_because_existing_outputs": skipped,
    }]))


# ============================================================
# 2. LIMPIEZA Y DATASETS
# ============================================================

def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    duplicated_cols = pd.Series(df.columns)[pd.Series(df.columns).duplicated()].tolist()
    if duplicated_cols:
        raise ValueError(f"Hay columnas duplicadas tras limpiar nombres: {duplicated_cols}")
    return df


def coerce_binary_target(series, target_col):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.Series(s.dropna().unique()).tolist())
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            return s.astype(int)

    mapping = {
        "0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1,
        "negative": 0, "positive": 1, "negativo": 0, "positivo": 1,
    }
    mapped = s.astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any():
        raise ValueError(f"No se puede convertir {target_col} a 0/1. Valores: {sorted(s.dropna().unique().tolist())}")
    return mapped.astype(int)


def load_and_clean_real_dataset(print_terminal_summary=True):
    if not DATA_FILE_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {DATA_FILE_NAME}. Colócalo en la misma carpeta que este script.")

    raw_excel_df = pd.read_excel(DATA_FILE_PATH, sheet_name=EXCEL_SHEET_NAME)
    raw_df = clean_column_names(raw_excel_df)
    raw_df = raw_df.replace(r"^\s*$", np.nan, regex=True)

    original_rows, original_cols = raw_df.shape

    unnamed_cols = [c for c in raw_df.columns if str(c).startswith("Unnamed")]
    raw_df = raw_df.drop(columns=unnamed_cols, errors="ignore")
    before_cols_empty = raw_df.shape[1]
    raw_df = raw_df.dropna(axis=1, how="all")
    before_rows_empty = raw_df.shape[0]
    raw_df = raw_df.dropna(axis=0, how="all").reset_index(drop=True)
    before_dups = len(raw_df)
    raw_df = raw_df.drop_duplicates().reset_index(drop=True)

    for target_col in TARGET_COLS:
        if target_col not in raw_df.columns:
            raise ValueError(f"No existe la columna objetivo {target_col}")
        raw_df[target_col] = coerce_binary_target(raw_df[target_col], target_col)

    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)

    all_risk_cols = [c for c in raw_df.columns if str(c).strip().startswith("Suicide_Risk")]
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != ID_COL and c not in all_risk_cols]
    constant_cols = [c for c in feature_cols if raw_df[c].nunique(dropna=True) <= 1]
    feature_cols = [c for c in feature_cols if c not in constant_cols]
    if not feature_cols:
        raise ValueError("No se han encontrado variables cuantitativas predictoras.")

    excluded_cols = []
    for col in raw_df.columns:
        if col == ID_COL:
            reason = "id_column_excluded"
        elif col in TARGET_COLS:
            reason = "target_active_analyzed"
        elif col in all_risk_cols:
            reason = "suicide_risk_target_excluded_not_analyzed"
        elif col in feature_cols:
            continue
        elif col in constant_cols:
            reason = "constant_numeric_column"
        else:
            reason = "excluded_because_not_quantitative"
        excluded_cols.append({
            "column": col,
            "reason": reason,
            "dtype": str(raw_df[col].dtype),
            "n_unique": int(raw_df[col].nunique(dropna=True)),
            "missing_count": int(raw_df[col].isna().sum()),
            "missing_pct": float(raw_df[col].isna().mean()),
        })

    clean_df_by_scenario = {}
    class_rows = []
    corr_rows = []
    cleaning_rows = []

    for scenario_name, config in SCENARIOS.items():
        target_col = config["target"]
        temp = raw_df[feature_cols + [target_col]].copy()
        temp["sample_id"] = np.arange(len(temp))

        n_before = len(temp)
        n_missing_x_rows = int(temp[feature_cols].isna().any(axis=1).sum())
        n_missing_x_total = int(temp[feature_cols].isna().sum().sum())
        n_missing_y = int(temp[target_col].isna().sum())

        clean_df = temp.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
        clean_df[target_col] = clean_df[target_col].astype(int)

        class_counts = clean_df[target_col].value_counts().sort_index()
        positives = int(class_counts.get(1, 0))
        negatives = int(class_counts.get(0, 0))
        if positives < 2 or negatives < 2:
            raise ValueError(f"{target_col} queda con menos de dos muestras en alguna clase.")
        if int(np.floor(positives * TEST_SIZE_FINAL)) < 1:
            raise ValueError(f"{target_col} tiene pocos positivos ({positives}) para partición clásica.")

        clean_df_by_scenario[scenario_name] = clean_df

        for cls in [0, 1]:
            count = int(class_counts.get(cls, 0))
            class_rows.append({
                "scenario": scenario_name,
                "target": target_col,
                "class": cls,
                "count": count,
                "proportion": float(count / len(clean_df)),
            })

        for feature in feature_cols:
            corr = clean_df[[feature, target_col]].corr().iloc[0, 1] if clean_df[feature].nunique(dropna=True) > 1 else np.nan
            corr_rows.append({
                "scenario": scenario_name,
                "target": target_col,
                "feature": feature,
                "pearson_corr_with_target": corr,
            })

        cleaning_rows.append({
            "scenario": scenario_name,
            "target": target_col,
            "rows_before_dropna": n_before,
            "rows_with_missing_in_selected_features_before_dropna": n_missing_x_rows,
            "missing_values_in_selected_features_before_dropna": n_missing_x_total,
            "rows_missing_target": n_missing_y,
            "rows_removed_missing_selected_features_or_target": n_before - len(clean_df),
            "rows_final_model_dataset": len(clean_df),
            "positive_class_count": positives,
            "negative_class_count": negatives,
        })

        clean_df.to_csv(DATASETS_DIR / f"{scenario_name}_clean_quantitative_full.csv", index=False)

    class_df = pd.DataFrame(class_rows)
    corr_df = pd.DataFrame(corr_rows).sort_values(
        ["scenario", "pearson_corr_with_target"],
        key=lambda s: s.abs() if pd.api.types.is_numeric_dtype(s) else s,
        ascending=[True, False],
    )
    cleaning_df = pd.DataFrame(cleaning_rows)
    excluded_df = pd.DataFrame(excluded_cols)

    class_df.to_csv(ANALYSIS_DIR / "target_class_distribution_by_scenario.csv", index=False)
    corr_df.to_csv(ANALYSIS_DIR / "correlation_with_target_by_scenario.csv", index=False)
    cleaning_df.to_csv(ANALYSIS_DIR / "cleaning_summary_by_scenario.csv", index=False)
    excluded_df.to_csv(ANALYSIS_DIR / "excluded_columns_report.csv", index=False)

    pd.DataFrame({
        "feature": feature_cols,
        "dtype": [str(raw_df[c].dtype) for c in feature_cols],
        "missing_count": [int(raw_df[c].isna().sum()) for c in feature_cols],
        "missing_pct": [float(raw_df[c].isna().mean()) for c in feature_cols],
        "n_unique": [int(raw_df[c].nunique(dropna=True)) for c in feature_cols],
    }).to_csv(ANALYSIS_DIR / "selected_quantitative_features.csv", index=False)

    raw_df[feature_cols].describe().T.to_csv(ANALYSIS_DIR / "numeric_descriptive_statistics.csv")

    pd.DataFrame([
        {"step": "filas_originales_excel", "value": original_rows},
        {"step": "columnas_originales_excel", "value": original_cols},
        {"step": "columnas_unnamed_eliminadas", "value": len(unnamed_cols)},
        {"step": "columnas_completamente_vacias_eliminadas", "value": before_cols_empty - raw_df.shape[1]},
        {"step": "filas_completamente_vacias_eliminadas", "value": before_rows_empty - len(raw_df)},
        {"step": "duplicados_exactos_eliminados", "value": before_dups - len(raw_df)},
        {"step": "variables_predictoras_cuantitativas_seleccionadas", "value": len(feature_cols)},
        {"step": "estrategia_validacion", "value": "classic_train_val_test_stratified_72_18_10"},
        {"step": "semillas", "value": len(SEEDS)},
        {"step": "imputacion", "value": False},
    ]).to_csv(ANALYSIS_DIR / "cleaning_summary.csv", index=False)

    lines = [
        "ANÁLISIS INICIAL DEL DATASET REAL 2 - PARTICIÓN CLÁSICA",
        "=" * 80,
        f"Archivo leído: {DATA_FILE_PATH}",
        f"Filas originales del Excel: {original_rows}",
        f"Columnas originales del Excel: {original_cols}",
        f"Variables predictoras cuantitativas seleccionadas: {len(feature_cols)}",
        f"Variables predictoras usadas: {', '.join(feature_cols)}",
        f"Validación: partición estratificada TRAIN/VAL/TEST = {TRAIN_SIZE_FINAL:.0%}/{VAL_SIZE_FINAL:.0%}/{TEST_SIZE_FINAL:.0%}",
        f"Semillas: {len(SEEDS)}",
        "Imputación: NO",
        "Risk 3 y Risk 4 se excluyen como targets activos por tener muy pocos positivos para partición clásica fiable.",
        "",
        "Distribución de clases por target:",
    ]
    for _, row in class_df.iterrows():
        lines.append(f"  {row['target']} | Clase {int(row['class'])}: {int(row['count'])} muestras ({row['proportion'] * 100:.2f}%)")
    with open(ANALYSIS_DIR / "dataset_initial_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    save_dataset_analysis_figures(clean_df_by_scenario, feature_cols, class_df)

    if print_terminal_summary:
        print("\n" + "=" * 90)
        print("ANÁLISIS EXPLORATORIO BREVE · DATASET REAL 2 · PARTICIÓN CLÁSICA")
        print("=" * 90)
        print(f"Archivo leído: {DATA_FILE_NAME}")
        print(f"Targets activos: {', '.join(TARGET_COLS)}")
        print(f"Split: TRAIN/VAL/TEST = {TRAIN_SIZE_FINAL:.0%}/{VAL_SIZE_FINAL:.0%}/{TEST_SIZE_FINAL:.0%}")
        print(f"Semillas: {len(SEEDS)}")
        print(f"Variables predictoras cuantitativas: {len(feature_cols)}")
        print("Variables:", ", ".join(feature_cols))
        print("\nDistribución de clases:")
        for scenario_name in SCENARIOS:
            temp = class_df[class_df["scenario"] == scenario_name]
            print(f"  - {scenario_name}:")
            for _, row in temp.iterrows():
                print(f"      Clase {int(row['class'])}: {int(row['count'])} ({row['proportion'] * 100:.2f}%)")
        print("\nTop 5 correlaciones absolutas con cada target:")
        for scenario_name in SCENARIOS:
            temp = corr_df[corr_df["scenario"] == scenario_name].copy()
            temp = temp.dropna(subset=["pearson_corr_with_target"])
            temp["abs_corr"] = temp["pearson_corr_with_target"].abs()
            temp = temp.sort_values("abs_corr", ascending=False).head(5)
            print(f"  - {SCENARIOS[scenario_name]['target']}:")
            for _, row in temp.iterrows():
                print(f"      {row['feature']}: corr={row['pearson_corr_with_target']:.4f}")
        print("=" * 90 + "\n")

    return raw_df, feature_cols, clean_df_by_scenario, excluded_cols


def save_dataset_analysis_figures(clean_df_by_scenario, feature_cols, class_distribution_df):
    for scenario_name, clean_df in clean_df_by_scenario.items():
        target_col = SCENARIOS[scenario_name]["target"]
        fig_dir = scenario_figures_dir(scenario_name)
        temp = class_distribution_df[class_distribution_df["scenario"] == scenario_name].copy()

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        bars = ax.bar([f"Clase {int(c)}" for c in temp["class"]], temp["count"].values, color=["#1f77b4", "#d62728"], edgecolor="white")
        ymax = max(temp["count"].max(), 1)
        for bar, (_, row) in zip(bars, temp.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(2.0, 0.025 * ymax),
                f"{int(row['count'])}\n{row['proportion'] * 100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(0, ymax * 1.22)
        ax.set_title(f"Dataset Real 2 · Distribución de clases · {pretty_target_label(target_col)}", pad=18)
        ax.set_ylabel("Número de muestras")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(fig_dir / f"{scenario_name}_target_distribution.png", dpi=180)
        plt.close(fig)



def seed_dataset_path(scenario_name, seed):
    path = DATASETS_DIR / scenario_short(scenario_name)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{scenario_short(scenario_name)}_seed{int(seed):02d}.csv"


def create_or_load_seed_dataset(seed, scenario_name, clean_df_by_scenario, feature_cols):
    path = seed_dataset_path(scenario_name, seed)
    target_col = SCENARIOS[scenario_name]["target"]
    required_cols = set(feature_cols + [target_col, "sample_id", "split"])

    if path.exists():
        df = pd.read_csv(path)
        if required_cols.issubset(df.columns):
            return df
        path.unlink()

    clean_df = clean_df_by_scenario[scenario_name].copy()
    y = clean_df[target_col].to_numpy(dtype=int)
    indices = np.arange(len(clean_df))

    idx_train_val, idx_test, y_train_val, _ = train_test_split(
        indices,
        y,
        test_size=TEST_SIZE_FINAL,
        random_state=seed,
        stratify=y,
    )

    val_relative_size = VAL_SIZE_FINAL / (TRAIN_SIZE_FINAL + VAL_SIZE_FINAL)
    idx_train, idx_val, _, _ = train_test_split(
        idx_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=seed,
        stratify=y_train_val,
    )

    df_train = clean_df.iloc[idx_train].copy()
    df_val = clean_df.iloc[idx_val].copy()
    df_test = clean_df.iloc[idx_test].copy()
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def build_all_seed_datasets(clean_df_by_scenario, feature_cols):
    rows = []
    for scenario_name in SCENARIOS:
        target_col = SCENARIOS[scenario_name]["target"]
        for seed in SEEDS:
            df = create_or_load_seed_dataset(seed, scenario_name, clean_df_by_scenario, feature_cols)
            for split_name in ["train", "val", "test"]:
                temp = df[df["split"] == split_name]
                counts = temp[target_col].value_counts().to_dict()
                rows.append({
                    "scenario": scenario_name,
                    "target": target_col,
                    "seed": seed,
                    "split": split_name,
                    "n_samples": int(len(temp)),
                    "class0": int(counts.get(0, 0)),
                    "class1": int(counts.get(1, 0)),
                })
    split_df = pd.DataFrame(rows)
    split_df.to_csv(ANALYSIS_DIR / "split_class_distribution_by_seed.csv", index=False)

    print("\nParticiones estratificadas generadas/reutilizadas:")
    summary = split_df.groupby(["scenario", "split"])[["n_samples", "class0", "class1"]].agg(["mean", "std"]).reset_index()
    print(summary.to_string(index=False))
    return split_df


# ============================================================
# 2.1. WARM-UP TÉCNICO
# ============================================================

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
    Warm-up técnico de joblib, StandardScaler, LogisticRegression y BLAS.
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


def preload_datasets_for_timing(clean_df_by_scenario, feature_cols):
    """
    Precarga las particiones guardadas antes de medir los bloques principales.

    Evita que la primera estrategia medida cargue en solitario el coste de
    lectura/generación de datasets o caché del sistema operativo. No se incluye
    en el coste comparable porque es calentamiento técnico común.
    """
    t_start = time.perf_counter()
    n_loaded = 0
    try:
        for scenario_name in SCENARIOS:
            target_col = SCENARIOS[scenario_name]["target"]
            for seed in SEEDS:
                df = create_or_load_seed_dataset(seed, scenario_name, clean_df_by_scenario, feature_cols)
                train_df = df[df["split"] == "train"].reset_index(drop=True)
                if not train_df.empty and feature_cols:
                    X_train = train_df[feature_cols].to_numpy(dtype=float)
                    y_train = train_df[target_col].to_numpy(dtype=int)
                    _ = StandardScaler().fit_transform(X_train)
                    _ = np.bincount(y_train.astype(int), minlength=2)
                n_loaded += 1

        elapsed = time.perf_counter() - t_start
        print(f"Precarga técnica de datasets terminada: {n_loaded} splits/semillas · {format_seconds(elapsed)}. No se incluye en el coste comparable.")
    except Exception as exc:
        print(f"Precarga técnica de datasets omitida por error no crítico: {exc}")
        print("El script continúa, pero el primer bloque medido podría incluir algo de sobrecoste inicial.")


def run_parallel_warmup(clean_df_by_scenario, feature_cols):
    """
    Warm-up técnico previo a los bloques medidos.

    Precarga datasets e inicializa pandas, NumPy/BLAS, métricas, joblib y
    Scikit-Learn. Este tiempo no se guarda como coste comparable.
    """
    print("\nWarm-up técnico: precargando datasets e inicializando pandas, NumPy, métricas, joblib y sklearn.")
    preload_datasets_for_timing(clean_df_by_scenario, feature_cols)
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


# ============================================================
# 3. COEFICIENTES Y ENTRENAMIENTO
# ============================================================

def grid_coeff_path(scenario_name):
    # Igual que en los códigos reales anteriores: carpeta por estrategia y un CSV wide por escenario.
    return GRID_MANUAL_COEFFICIENTS_WIDE_DIR / f"{scenario_short(scenario_name)}_grid_manual_coeffs.csv"


def final_coeff_path(scenario_name, model_name):
    # No existe carpeta final_models: equal_weights y sklearn_balanced se guardan como estrategias independientes.
    # Best F1 se evalúa directamente desde grid_manual usando el par (c0, c1) seleccionado.
    if model_name == "equal_weights":
        return EQUAL_WEIGHTS_COEFFICIENTS_WIDE_DIR / f"{scenario_short(scenario_name)}_equal_weights_coeffs.csv"
    if model_name == "sklearn_balanced":
        return SKLEARN_BALANCED_COEFFICIENTS_WIDE_DIR / f"{scenario_short(scenario_name)}_sklearn_balanced_coeffs.csv"
    raise ValueError(f"Los coeficientes de {model_name} no se guardan en carpeta final independiente. Para Best F1 se usa la rejilla manual guardada.")


def coefficient_row(scenario_name, seed, model_name, c0, c1, model, feature_cols, conv):
    row = {
        "scenario": scenario_name,
        "seed": int(seed),
        "model_name": model_name,
        "c0": float(c0),
        "c1": float(c1),
        "intercept": float(model.intercept_[0]),
        "n_iter": conv["n_iter"],
        "max_iter": conv["max_iter"],
        "convergence_warning": conv["convergence_warning"],
        "reached_max_iter": conv["reached_max_iter"],
        "converged_without_warning": conv["converged_without_warning"],
        "convergence_message": conv["convergence_message"],
    }
    for name, beta in zip(feature_cols, model.coef_[0]):
        row[name] = float(beta)
    return row


def coefficients_file_ok(path, scenario_name, feature_cols, model_names, rows_per_seed=None):
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    required = {"scenario", "seed", "model_name", "c0", "c1", "intercept", "n_iter", "max_iter"} | set(feature_cols)
    if not required.issubset(df.columns):
        return False
    if set(df["scenario"].astype(str).unique()) != {str(scenario_name)}:
        return False
    if set(df["seed"].astype(int).unique()) != set(SEEDS):
        return False
    if set(df["model_name"].astype(str).unique()) != set(model_names):
        return False
    if rows_per_seed is not None:
        counts = df.groupby("seed").size()
        if not np.all(counts.to_numpy(dtype=int) == rows_per_seed):
            return False
    return True


def train_grid_one_seed(seed, scenario_name, clean_df_by_scenario, feature_cols):
    target_col = SCENARIOS[scenario_name]["target"]
    start = time.perf_counter()

    df = create_or_load_seed_dataset(seed, scenario_name, clean_df_by_scenario, feature_cols)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rows = []
    warnings_total = 0
    max_iter_total = 0

    for c0 in C_VALUES:
        for c1 in C_VALUES:
            model = make_logistic_model({0: c0, 1: c1}, seed)
            conv = fit_model_with_convergence_info(model, X_train_scaled, y_train)
            warnings_total += int(conv["convergence_warning"])
            max_iter_total += int(conv["reached_max_iter"])
            rows.append(coefficient_row(scenario_name, seed, GRID_MODEL_NAME, c0, c1, model, feature_cols, conv))

    timer = {
        "scenario": scenario_name,
        "target": target_col,
        "seed": seed,
        "stage": "grid_train_seed",
        "model_name": GRID_MODEL_NAME,
        "n_models": len(C_VALUES) * len(C_VALUES),
        "convergence_warnings": warnings_total,
        "reached_max_iter_count": max_iter_total,
        "total_seconds": time.perf_counter() - start,
        "total_time_readable": format_seconds(time.perf_counter() - start),
    }

    return pd.DataFrame(rows), pd.DataFrame([timer])


def train_or_load_grid(clean_df_by_scenario, feature_cols, timer_frames):
    expected_rows = len(C_VALUES) * len(C_VALUES)

    for scenario_name in SCENARIOS:
        path = grid_coeff_path(scenario_name)
        if coefficients_file_ok(path, scenario_name, feature_cols, {GRID_MODEL_NAME}, rows_per_seed=expected_rows):
            t = time.perf_counter()
            print(f"\nCoeficientes de rejilla existentes compatibles: {scenario_name}. Se reutilizan.")
            add_wall_clock_timer(timer_frames, scenario_name, "load_existing_grid_coefficients", t, n_models=0, skipped=True)
            continue

        start = time.perf_counter()
        print("\n" + "=" * 90)
        print(f"TRAIN · REJILLA MANUAL · ESCENARIO: {scenario_name.upper()}")
        print("=" * 90)
        print(f"Semillas: {len(SEEDS)}")
        print(f"Rejilla: {len(C_VALUES)} x {len(C_VALUES)} = {expected_rows} modelos por seed")
        print("Paralelización: solo se paralelizan las semillas.")

        outputs = Parallel(n_jobs=N_JOBS)(
            delayed(train_grid_one_seed)(seed, scenario_name, clean_df_by_scenario, feature_cols)
            for seed in SEEDS
        )
        coeff_df = pd.concat([x[0] for x in outputs], ignore_index=True)
        timer_df = pd.concat([x[1] for x in outputs], ignore_index=True)
        timer_frames.append(timer_df)

        coeff_df = coeff_df.sort_values(["scenario", "seed", "c0", "c1"]).reset_index(drop=True)
        coeff_df.to_csv(path, index=False)

        n_models = int(timer_df["n_models"].sum())
        warnings_total = int(timer_df["convergence_warnings"].sum())
        max_iter_total = int(timer_df["reached_max_iter_count"].sum())
        print(f"Convergencia TRAIN rejilla {scenario_name}: warnings={warnings_total}/{n_models}, alcanzan max_iter={max_iter_total}/{n_models}")

        add_wall_clock_timer(
            timer_frames, scenario_name, "grid_train_scenario_parallel_total",
            start, n_models=len(SEEDS) * expected_rows, skipped=False
        )


def load_seed_coeffs(path, seed):
    df = pd.read_csv(path)
    return df[df["seed"].astype(int) == int(seed)].reset_index(drop=True)


def extract_coefficients(seed_coeffs, feature_cols, c0=None, c1=None):
    temp = seed_coeffs
    if c0 is not None and c1 is not None:
        mask = np.isclose(temp["c0"].astype(float), float(c0)) & np.isclose(temp["c1"].astype(float), float(c1))
        temp = temp[mask]
    if temp.empty:
        return None
    row = temp.iloc[0]
    conv = {
        "n_iter": row.get("n_iter", np.nan),
        "max_iter": row.get("max_iter", np.nan),
        "convergence_warning": bool(row.get("convergence_warning", False)),
        "reached_max_iter": bool(row.get("reached_max_iter", False)),
        "converged_without_warning": row.get("converged_without_warning", np.nan),
        "convergence_message": row.get("convergence_message", ""),
    }
    return float(row["intercept"]), row[feature_cols].to_numpy(dtype=float), conv, row


# ============================================================
# 4. VALIDACIÓN DE REJILLA Y SELECCIÓN
# ============================================================

def add_prediction_rows(pred_rows, scenario_name, seed, model_name, split_name, eval_df, y_true, y_pred, y_prob, c0, c1):
    target_col = SCENARIOS[scenario_name]["target"]
    for sample_id, yt, yp, prob in zip(eval_df["sample_id"], y_true, y_pred, y_prob):
        pred_rows.append({
            "scenario": scenario_name,
            "target": target_col,
            "seed": int(seed),
            "model_name": model_name,
            "split": split_name,
            "sample_id": int(sample_id),
            "c0": float(c0),
            "c1": float(c1),
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_prob": float(prob),
        })


def evaluate_grid_one_seed(seed, scenario_name, clean_df_by_scenario, feature_cols, split_name):
    target_col = SCENARIOS[scenario_name]["target"]
    start = time.perf_counter()

    df = create_or_load_seed_dataset(seed, scenario_name, clean_df_by_scenario, feature_cols)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    eval_df = df[df["split"] == split_name].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_eval = eval_df[feature_cols].to_numpy(dtype=float)
    y_eval = eval_df[target_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    seed_coeffs = load_seed_coeffs(grid_coeff_path(scenario_name), seed)

    result_rows = []
    prediction_rows = []
    warnings_total = 0
    max_iter_total = 0

    for c0 in C_VALUES:
        for c1 in C_VALUES:
            extracted = extract_coefficients(seed_coeffs, feature_cols, c0, c1)
            if extracted is None:
                raise FileNotFoundError(f"Faltan coeficientes para {scenario_name}, seed={seed}, c0={c0}, c1={c1}")
            intercept, betas, conv, _ = extracted
            warnings_total += int(conv["convergence_warning"])
            max_iter_total += int(conv["reached_max_iter"])

            y_prob, y_pred = predict_from_saved_coefficients(X_eval_scaled, intercept, betas)
            metrics = compute_metrics_for_split(y_eval, y_pred, y_prob, split_name)

            row = {
                "scenario": scenario_name,
                "target": target_col,
                "seed": int(seed),
                "model_name": GRID_MODEL_NAME,
                "c0": float(c0),
                "c1": float(c1),
                "n_iter": conv["n_iter"],
                "max_iter": conv["max_iter"],
                "convergence_warning": conv["convergence_warning"],
                "reached_max_iter": conv["reached_max_iter"],
                "converged_without_warning": conv["converged_without_warning"],
                "convergence_message": conv["convergence_message"],
                "used_saved_coefficients": True,
            }
            row.update(metrics)
            result_rows.append(row)
            add_prediction_rows(prediction_rows, scenario_name, seed, GRID_MODEL_NAME, split_name, eval_df, y_eval, y_pred, y_prob, c0, c1)

    timer = {
        "scenario": scenario_name,
        "target": target_col,
        "seed": seed,
        "stage": f"grid_{split_name}_evaluation_seed",
        "model_name": GRID_MODEL_NAME,
        "n_models": len(C_VALUES) * len(C_VALUES),
        "convergence_warnings": warnings_total,
        "reached_max_iter_count": max_iter_total,
        "total_seconds": time.perf_counter() - start,
        "total_time_readable": format_seconds(time.perf_counter() - start),
        "used_saved_coefficients": True,
    }

    return pd.DataFrame(result_rows), pd.DataFrame(prediction_rows), pd.DataFrame([timer])


def aggregate_results(raw_df, group_cols):
    metric_cols = [
        col for col in raw_df.columns
        if col not in ["scenario", "target", "seed", "model_name", "convergence_message", "used_saved_coefficients"]
        and pd.api.types.is_numeric_dtype(raw_df[col])
    ]
    agg = raw_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    new_cols = []
    for col in agg.columns:
        if col[1] == "":
            new_cols.append(col[0])
        else:
            new_cols.append(f"{col[0]}_{col[1]}")
    agg.columns = new_cols
    std_cols = [c for c in agg.columns if c.endswith("_std")]
    agg[std_cols] = agg[std_cols].fillna(0.0)
    return agg



def compute_balanced_points(clean_df_by_scenario, feature_cols):
    rows = []
    for scenario_name in SCENARIOS:
        target_col = SCENARIOS[scenario_name]["target"]
        for seed in SEEDS:
            df = create_or_load_seed_dataset(seed, scenario_name, clean_df_by_scenario, feature_cols)
            y_train = df[df["split"] == "train"][target_col].to_numpy(dtype=int)
            c0, c1 = compute_sklearn_balanced_weights(y_train)
            rows.append({"scenario": scenario_name, "target": target_col, "seed": int(seed), "balanced_c0": c0, "balanced_c1": c1})
    raw = pd.DataFrame(rows)
    raw.to_csv(PATHS["balanced_raw"], index=False)

    agg = raw.groupby(["scenario", "target"])[["balanced_c0", "balanced_c1"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["scenario", "target", "balanced_c0_mean", "balanced_c0_std", "balanced_c1_mean", "balanced_c1_std"]
    agg.to_csv(PATHS["balanced"], index=False)
    return agg


def evaluate_grid_validation(clean_df_by_scenario, feature_cols, timer_frames):
    train_or_load_grid(clean_df_by_scenario, feature_cols, timer_frames)

    all_results = []
    all_predictions = []

    for scenario_name in SCENARIOS:
        start = time.perf_counter()
        print("\n" + "=" * 90)
        print(f"VALIDACIÓN · REJILLA MANUAL · ESCENARIO: {scenario_name.upper()}")
        print("=" * 90)
        print("No se entrena: se cargan coeficientes guardados en TRAIN y se evalúa VAL.")

        outputs = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_grid_one_seed)(seed, scenario_name, clean_df_by_scenario, feature_cols, "val")
            for seed in SEEDS
        )
        result_df = pd.concat([x[0] for x in outputs], ignore_index=True)
        pred_df = pd.concat([x[1] for x in outputs], ignore_index=True)
        timer_df = pd.concat([x[2] for x in outputs], ignore_index=True)

        all_results.append(result_df)
        all_predictions.append(pred_df)
        timer_frames.append(timer_df)

        add_wall_clock_timer(
            timer_frames, scenario_name, "grid_val_evaluation_scenario_parallel_total",
            start, n_models=len(SEEDS) * len(C_VALUES) * len(C_VALUES), skipped=False
        )

    raw = pd.concat(all_results, ignore_index=True).sort_values(["scenario", "seed", "c0", "c1"]).reset_index(drop=True)
    pred = pd.concat(all_predictions, ignore_index=True).sort_values(["scenario", "seed", "c0", "c1", "sample_id"]).reset_index(drop=True)

    raw.to_csv(PATHS["val_grid_raw"], index=False)
    pred.to_csv(PATHS["val_grid_pred"], index=False)

    # Igual que en los otros códigos: la agregación mean/std y la selección Best F1
    # se miden por escenario y se muestran desglosadas en el resumen de tiempos.
    agg_frames = []
    best_rows = []

    for scenario_name in SCENARIOS:
        start_agg = time.perf_counter()
        scenario_raw = raw[raw["scenario"] == scenario_name].copy()
        scenario_agg = aggregate_results(scenario_raw, ["scenario", "target", "c0", "c1"])
        scenario_agg = scenario_agg.sort_values(["scenario", "c0", "c1"]).reset_index(drop=True)
        agg_frames.append(scenario_agg)
        add_wall_clock_timer(
            timer_frames,
            scenario_name,
            "grid_aggregation_by_scenario",
            start_agg,
            n_models=len(scenario_raw),
            skipped=False,
        )

        start_sel = time.perf_counter()
        idx_best = scenario_agg["val_f1_mean"].idxmax()
        best_rows.append(scenario_agg.loc[idx_best])
        add_wall_clock_timer(
            timer_frames,
            scenario_name,
            "grid_best_selection_by_scenario",
            start_sel,
            n_models=len(scenario_agg),
            skipped=False,
        )

    agg = pd.concat(agg_frames, ignore_index=True).sort_values(["scenario", "c0", "c1"]).reset_index(drop=True)
    best = pd.DataFrame(best_rows).reset_index(drop=True)

    agg.to_csv(PATHS["val_grid_agg"], index=False)
    best.to_csv(PATHS["best_configs"], index=False)
    balanced = compute_balanced_points(clean_df_by_scenario, feature_cols)

    return raw, pred, agg, best, balanced


# ============================================================
# 5. MODELOS FINALES
# ============================================================

def train_final_one_seed(seed, scenario_name, clean_df_by_scenario, feature_cols, model_name):
    target_col = SCENARIOS[scenario_name]["target"]
    start = time.perf_counter()

    df = create_or_load_seed_dataset(seed, scenario_name, clean_df_by_scenario, feature_cols)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if model_name == "equal_weights":
        class_weight = {0: EQUAL_WEIGHTS_C0, 1: EQUAL_WEIGHTS_C1}
        c0, c1 = EQUAL_WEIGHTS_C0, EQUAL_WEIGHTS_C1
    elif model_name == "sklearn_balanced":
        class_weight = "balanced"
        c0, c1 = compute_sklearn_balanced_weights(y_train)
    else:
        raise ValueError(model_name)

    model = make_logistic_model(class_weight, seed)
    conv = fit_model_with_convergence_info(model, X_train_scaled, y_train)
    row = coefficient_row(scenario_name, seed, model_name, c0, c1, model, feature_cols, conv)

    timer = {
        "scenario": scenario_name,
        "target": target_col,
        "seed": int(seed),
        "stage": "final_train_seed",
        "model_name": model_name,
        "n_models": 1,
        "convergence_warnings": int(conv["convergence_warning"]),
        "reached_max_iter_count": int(conv["reached_max_iter"]),
        "total_seconds": time.perf_counter() - start,
        "total_time_readable": format_seconds(time.perf_counter() - start),
    }
    return pd.DataFrame([row]), pd.DataFrame([timer])


def final_coefficients_ok(path, scenario_name, model_name, feature_cols):
    return coefficients_file_ok(path, scenario_name, feature_cols, {model_name}, rows_per_seed=1)


def train_or_load_final_coefficients(clean_df_by_scenario, feature_cols, best_df, timer_frames):
    """
    Genera o reutiliza los coeficientes wide independientes de equal_weights y
    sklearn_balanced.

    El modelo Manual Best F1 no se guarda en una carpeta aparte: se evalúa
    cargando directamente los coeficientes de la rejilla manual del punto
    seleccionado, igual que en el código sintético base.
    """
    _ = best_df  # Se mantiene en la firma para conservar el flujo general.

    for scenario_name in SCENARIOS:
        print(f"\nCoeficientes Best F1: {scenario_name}. Se usarán desde la rejilla manual guardada.")

        for model_name in ["equal_weights", "sklearn_balanced"]:
            path = final_coeff_path(scenario_name, model_name)
            if final_coefficients_ok(path, scenario_name, model_name, feature_cols):
                start = time.perf_counter()
                print(f"Coeficientes wide {model_name} compatibles: {scenario_name}. Se reutilizan.")
                add_wall_clock_timer(timer_frames, scenario_name, f"load_existing_final_{model_name}_coefficients", start, n_models=0, skipped=True)
                continue

            start = time.perf_counter()
            print("\n" + "=" * 90)
            print(f"TRAIN · {model_name.upper()} · ESCENARIO: {scenario_name.upper()}")
            print("=" * 90)
            print("Se entrena 1 modelo por seed y se guardan coeficientes wide en carpeta independiente.")
            print(f"Carpeta destino: {path.parent}")

            outputs = Parallel(n_jobs=N_JOBS)(
                delayed(train_final_one_seed)(seed, scenario_name, clean_df_by_scenario, feature_cols, model_name)
                for seed in SEEDS
            )
            coeff_df = pd.concat([x[0] for x in outputs], ignore_index=True)
            timer_df = pd.concat([x[1] for x in outputs], ignore_index=True)
            timer_frames.append(timer_df)

            coeff_df = coeff_df.sort_values(["scenario", "seed", "model_name"]).reset_index(drop=True)
            coeff_df.to_csv(path, index=False)

            n_models = int(timer_df["n_models"].sum())
            warnings_total = int(timer_df["convergence_warnings"].sum())
            max_iter_total = int(timer_df["reached_max_iter_count"].sum())
            print(f"Convergencia TRAIN {model_name} {scenario_name}: warnings={warnings_total}/{n_models}, alcanzan max_iter={max_iter_total}/{n_models}")
            print(f"Coeficientes guardados en: {path}")

            add_wall_clock_timer(timer_frames, scenario_name, f"final_{model_name}_train_scenario_parallel_total", start, n_models=len(SEEDS), skipped=False)


def evaluate_final_one_seed(seed, scenario_name, clean_df_by_scenario, feature_cols, model_name, split_name, best_c0=None, best_c1=None):
    target_col = SCENARIOS[scenario_name]["target"]
    start = time.perf_counter()

    df = create_or_load_seed_dataset(seed, scenario_name, clean_df_by_scenario, feature_cols)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    eval_df = df[df["split"] == split_name].reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_eval = eval_df[feature_cols].to_numpy(dtype=float)
    y_eval = eval_df[target_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    if model_name == "best_val_f1_grid":
        if best_c0 is None or best_c1 is None:
            raise ValueError("Para evaluar best_val_f1_grid deben pasarse best_c0 y best_c1.")
        seed_coeffs = load_seed_coeffs(grid_coeff_path(scenario_name), seed)
        extracted = extract_coefficients(seed_coeffs, feature_cols, best_c0, best_c1)
        if extracted is None:
            raise FileNotFoundError(f"Faltan coeficientes de rejilla para Best F1, {scenario_name}, seed={seed}, c0={best_c0}, c1={best_c1}")
        intercept, betas, conv, coeff_row = extracted
        c0_eval, c1_eval = float(best_c0), float(best_c1)
    else:
        seed_coeffs = load_seed_coeffs(final_coeff_path(scenario_name, model_name), seed)
        extracted = extract_coefficients(seed_coeffs, feature_cols)
        if extracted is None:
            raise FileNotFoundError(f"Faltan coeficientes finales {model_name}, {scenario_name}, seed={seed}")
        intercept, betas, conv, coeff_row = extracted
        c0_eval, c1_eval = float(coeff_row["c0"]), float(coeff_row["c1"])

    y_prob, y_pred = predict_from_saved_coefficients(X_eval_scaled, intercept, betas)
    metrics = compute_metrics_for_split(y_eval, y_pred, y_prob, split_name)

    result = {
        "scenario": scenario_name,
        "target": target_col,
        "seed": int(seed),
        "model_name": model_name,
        "c0": c0_eval,
        "c1": c1_eval,
        "n_iter": conv["n_iter"],
        "max_iter": conv["max_iter"],
        "convergence_warning": conv["convergence_warning"],
        "reached_max_iter": conv["reached_max_iter"],
        "converged_without_warning": conv["converged_without_warning"],
        "convergence_message": conv["convergence_message"],
        "used_saved_coefficients": True,
    }
    result.update(metrics)

    prediction_rows = []
    add_prediction_rows(prediction_rows, scenario_name, seed, model_name, split_name, eval_df, y_eval, y_pred, y_prob, c0_eval, c1_eval)

    timer = {
        "scenario": scenario_name,
        "target": target_col,
        "seed": int(seed),
        "stage": f"final_{split_name}_evaluation_seed",
        "model_name": model_name,
        "n_models": 1,
        "convergence_warnings": int(conv["convergence_warning"]),
        "reached_max_iter_count": int(conv["reached_max_iter"]),
        "total_seconds": time.perf_counter() - start,
        "total_time_readable": format_seconds(time.perf_counter() - start),
        "used_saved_coefficients": True,
    }
    return pd.DataFrame([result]), pd.DataFrame(prediction_rows), pd.DataFrame([timer])


def make_comparison_table(agg_df, split_name):
    cols = [
        "scenario", "target", "model_name", "c0_mean", "c0_std", "c1_mean", "c1_std",
        f"{split_name}_f1_mean", f"{split_name}_f1_std",
        f"{split_name}_pr_auc_mean", f"{split_name}_pr_auc_std",
        f"{split_name}_balanced_accuracy_mean", f"{split_name}_balanced_accuracy_std",
        f"{split_name}_mcc_mean", f"{split_name}_mcc_std",
        f"{split_name}_recall_mean", f"{split_name}_recall_std",
        f"{split_name}_fnr_mean", f"{split_name}_fnr_std",
        f"{split_name}_specificity_mean", f"{split_name}_specificity_std",
        f"{split_name}_roc_auc_mean", f"{split_name}_roc_auc_std",
        f"{split_name}_tn_mean", f"{split_name}_fp_mean", f"{split_name}_fn_mean", f"{split_name}_tp_mean",
    ]
    return agg_df[[c for c in cols if c in agg_df.columns]].copy()


def save_final_outputs(split_name, raw, pred, agg, comp):
    raw.to_csv(PATHS[f"{split_name}_final_raw"], index=False)
    pred.to_csv(PATHS[f"{split_name}_final_pred"], index=False)
    agg.to_csv(PATHS[f"{split_name}_final_agg"], index=False)
    comp.to_csv(PATHS[f"{split_name}_final_comp"], index=False)


def evaluate_final_split(clean_df_by_scenario, feature_cols, best_df, split_name, timer_frames):
    train_or_load_final_coefficients(clean_df_by_scenario, feature_cols, best_df, timer_frames)

    all_results = []
    all_predictions = []

    for scenario_name in SCENARIOS:
        best_row = best_df[best_df["scenario"] == scenario_name].iloc[0]
        best_c0 = float(best_row["c0"])
        best_c1 = float(best_row["c1"])

        for model_name in FINAL_MODEL_NAMES:
            start = time.perf_counter()
            print("\n" + "=" * 90)
            print(f"{split_name.upper()} · COMPARACIÓN FINAL · {model_name.upper()} · ESCENARIO: {scenario_name.upper()}")
            print("=" * 90)
            if model_name == "best_val_f1_grid":
                print(f"Best F1 usa coeficientes de la rejilla manual ya guardada: c0={best_c0}, c1={best_c1}.")
            else:
                print(f"{model_name} usa su archivo wide independiente guardado en coefficients_wide/{model_name}/.")
            print("No se entrena: se cargan coeficientes finales guardados por seed.")

            outputs = Parallel(n_jobs=N_JOBS)(
                delayed(evaluate_final_one_seed)(
                    seed,
                    scenario_name,
                    clean_df_by_scenario,
                    feature_cols,
                    model_name,
                    split_name,
                    best_c0,
                    best_c1,
                )
                for seed in SEEDS
            )

            result_df = pd.concat([x[0] for x in outputs], ignore_index=True)
            pred_df = pd.concat([x[1] for x in outputs], ignore_index=True)
            timer_df = pd.concat([x[2] for x in outputs], ignore_index=True)

            all_results.append(result_df)
            all_predictions.append(pred_df)
            timer_frames.append(timer_df)

            add_wall_clock_timer(timer_frames, scenario_name, f"final_{model_name}_{split_name}_evaluation_scenario_parallel_total", start, n_models=len(SEEDS), skipped=False)

    raw = pd.concat(all_results, ignore_index=True)
    pred = pd.concat(all_predictions, ignore_index=True)

    rank = {name: idx for idx, name in enumerate(FINAL_MODEL_NAMES)}
    raw["_rank"] = raw["model_name"].map(rank)
    raw = raw.sort_values(["scenario", "_rank", "seed"]).drop(columns="_rank").reset_index(drop=True)
    pred["_rank"] = pred["model_name"].map(rank)
    pred = pred.sort_values(["scenario", "_rank", "seed", "sample_id"]).drop(columns="_rank").reset_index(drop=True)

    agg = aggregate_results(raw, ["scenario", "target", "model_name"])
    agg["_rank"] = agg["model_name"].map(rank)
    agg = agg.sort_values(["scenario", "_rank"]).drop(columns="_rank").reset_index(drop=True)

    comp = make_comparison_table(agg, split_name)
    save_final_outputs(split_name, raw, pred, agg, comp)

    return raw, pred, agg, comp


# ============================================================
# 6. FIGURAS
# ============================================================

def pretty_target_label(target_name):
    """
    Convierte nombres técnicos de targets en etiquetas limpias para figuras.
    """
    label = str(target_name).replace("_", " ").strip()
    label = label.replace("Risk1", "Risk 1").replace("Risk2", "Risk 2")
    label = label.replace("Risk3", "Risk 3").replace("Risk4", "Risk 4")
    return label


def pretty_scenario_label(scenario_name):
    """
    Etiqueta legible de escenario para títulos y ejes de figuras.
    """
    label = str(scenario_name)
    label = label.replace("real_2_suicide_risk", "Risk ")
    label = " ".join(label.replace("_", " ").split())
    return label.title().replace("Risk 1", "Risk 1").replace("Risk 2", "Risk 2")


def get_scenario_figure_path(scenario_name, filename):
    return scenario_figures_dir(scenario_name) / filename


def get_global_figure_path(filename):
    GLOBAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return GLOBAL_FIGURES_DIR / filename


def format_split_label_for_plot(split_name):
    split_labels = {
        "val": "VALIDACIÓN",
        "test": "TEST",
        "train": "TRAIN",
        "warmup": "WARM-UP",
    }
    return split_labels.get(str(split_name).lower(), str(split_name).replace("_", " ").upper())


def format_stat_label_for_plot(stat_name):
    stat_labels = {
        "mean": "Mean",
        "std": "STD",
    }
    return stat_labels.get(str(stat_name).lower(), str(stat_name).replace("_", " ").title())


def format_base_metric_label_for_plot(base_metric):
    base_metric = str(base_metric)

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
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "tn": "TN",
        "fp": "FP",
        "fn": "FN",
        "tp": "TP",
    }

    if base_metric in labels:
        return labels[base_metric]

    return " ".join(part.capitalize() for part in base_metric.split("_"))


def parse_metric_name_for_plot(metric):
    """
    Separa nombres técnicos tipo val_balanced_accuracy_mean en:
    split=val, base_metric=balanced_accuracy, stat=mean.
    """
    parts = str(metric).split("_")

    split_name = None
    if parts and parts[0] in {"train", "val", "test", "warmup"}:
        split_name = parts[0]
        parts = parts[1:]

    stat_name = None
    if parts and parts[-1] in {"mean", "std"}:
        stat_name = parts[-1]
        parts = parts[:-1]

    base_metric = "_".join(parts)

    return split_name, base_metric, stat_name


def metric_label(metric, include_split=True, include_stat=True):
    """
    Etiqueta limpia para títulos, leyendas, ejes y barras de color.
    """
    split_name, base_metric, stat_name = parse_metric_name_for_plot(metric)

    label_parts = []

    if include_split and split_name is not None:
        label_parts.append(format_split_label_for_plot(split_name))

    label_parts.append(format_base_metric_label_for_plot(base_metric))

    if include_stat and stat_name is not None:
        label_parts.append(format_stat_label_for_plot(stat_name))

    return " · ".join(label_parts)


def get_model_plot_style(model_name):
    styles = {
        "best_val_f1_grid": {"label": "Manual (Best F1)", "short_label": "Manual", "color": MODEL_COLORS["best_val_f1_grid"]},
        "equal_weights": {"label": "Equal weights", "short_label": "Equal weights", "color": MODEL_COLORS["equal_weights"]},
        "sklearn_balanced": {"label": "Sklearn balanced", "short_label": "Sklearn balanced", "color": MODEL_COLORS["sklearn_balanced"]},
    }
    return styles.get(model_name, {"label": model_name, "short_label": model_name, "color": "#777777"})


def format_weight_for_bar(value):
    if pd.isna(value):
        return "nan"
    return f"{float(value):.3g}"


def value_to_axis_position(value, sorted_values):
    sorted_values = np.asarray(sorted_values, dtype=float)
    value = max(float(value), float(sorted_values[0]))
    log_values = np.log10(sorted_values)
    log_value = np.log10(value)
    if log_value <= log_values[0]:
        return 0.0
    if log_value >= log_values[-1]:
        return float(len(sorted_values) - 1)
    return float(np.interp(log_value, log_values, np.arange(len(sorted_values))))


def get_nearest_metric_value(pivot, c0, c1):
    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)
    nearest_c0 = c0_values[np.argmin(np.abs(np.log10(c0_values) - np.log10(float(c0))))]
    nearest_c1 = c1_values[np.argmin(np.abs(np.log10(c1_values) - np.log10(float(c1))))]
    value = pivot.loc[nearest_c0, nearest_c1]
    return 0.0 if pd.isna(value) else float(value)


def get_real_model_metric_value(comparison_df, scenario_name, model_name, metric):
    if comparison_df is None or metric not in comparison_df.columns:
        return None
    rows = comparison_df[(comparison_df["scenario"] == scenario_name) & (comparison_df["model_name"] == model_name)]
    if rows.empty or pd.isna(rows.iloc[0][metric]):
        return None
    return float(rows.iloc[0][metric])


def get_label_metric_value_for_pin(point, pivot, metric, scenario_name, val_comparison_df=None):
    if point.get("model_name") in ["sklearn_balanced", "equal_weights"]:
        real_value = get_real_model_metric_value(val_comparison_df, scenario_name, point.get("model_name"), metric)
        if real_value is not None:
            return real_value
    return get_nearest_metric_value(pivot, point["c0"], point["c1"])


def get_pin_points(best_df, balanced_df, scenario_name):
    best_row = best_df[best_df["scenario"] == scenario_name].iloc[0]
    balanced_row = balanced_df[balanced_df["scenario"] == scenario_name].iloc[0]

    best_point = {
        "label": "Best val F1",
        "short_label": "Best F1",
        "model_name": "best_val_f1_grid",
        "c0": float(best_row["c0"]),
        "c1": float(best_row["c1"]),
        "c0_std": 0.0,
        "c1_std": 0.0,
        "is_grid_point": True,
    }
    balanced_point = {
        "label": "Sklearn balanced",
        "short_label": "Sklearn balanced",
        "model_name": "sklearn_balanced",
        "c0": float(balanced_row["balanced_c0_mean"]),
        "c1": float(balanced_row["balanced_c1_mean"]),
        "c0_std": float(balanced_row.get("balanced_c0_std", 0.0) or 0.0),
        "c1_std": float(balanced_row.get("balanced_c1_std", 0.0) or 0.0),
        "is_grid_point": False,
    }
    equal_point = {
        "label": "Equal weights",
        "short_label": "Equal weights",
        "model_name": "equal_weights",
        "c0": EQUAL_WEIGHTS_C0,
        "c1": EQUAL_WEIGHTS_C1,
        "c0_std": 0.0,
        "c1_std": 0.0,
        "is_grid_point": any(np.isclose(np.asarray(C_VALUES, dtype=float), EQUAL_WEIGHTS_C0)) and any(np.isclose(np.asarray(C_VALUES, dtype=float), EQUAL_WEIGHTS_C1)),
    }
    return best_point, balanced_point, equal_point


def format_weights_for_pin_text(point):
    c0_std = abs(float(point.get("c0_std", 0.0) or 0.0))
    c1_std = abs(float(point.get("c1_std", 0.0) or 0.0))
    if c0_std > 1e-12 or c1_std > 1e-12:
        return f"c0_mean={point['c0']:.3g}, c1_mean={point['c1']:.3g}"
    return f"c0={point['c0']:.3g}, c1={point['c1']:.3g}"


def format_pin_text(point, metric_value):
    label = point.get("short_label", point.get("label", "Punto"))
    return f"{label}: {metric_value:.3f}\n{format_weights_for_pin_text(point)}"


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def estimate_label_box_size():
    grid_span = max(1.0, float(len(C_VALUES) - 1))
    box_width = clamp(0.29 * grid_span, 4.80, 7.20)
    box_height = clamp(0.075 * grid_span, 1.35, 1.90)
    return box_width, box_height


def make_label_rect(text_x, text_y, box_width, box_height):
    return (text_x, text_x + box_width, text_y - box_height / 2, text_y + box_height / 2)


def rectangles_overlap(rect_a, rect_b, padding=0.34):
    ax0, ax1, ay0, ay1 = rect_a
    bx0, bx1, by0, by1 = rect_b
    return not (ax1 + padding < bx0 or bx1 + padding < ax0 or ay1 + padding < by0 or by1 + padding < ay0)


def overlap_area(rect_a, rect_b):
    ax0, ax1, ay0, ay1 = rect_a
    bx0, bx1, by0, by1 = rect_b
    return max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))


def point_inside_rect(point_x, point_y, rect, padding=0.58):
    x0, x1, y0, y1 = rect
    return x0 - padding <= point_x <= x1 + padding and y0 - padding <= point_y <= y1 + padding


def nearest_point_on_rect(point_x, point_y, rect):
    x0, x1, y0, y1 = rect
    nearest_x = clamp(point_x, x0, x1)
    nearest_y = clamp(point_y, y0, y1)
    if x0 < point_x < x1 and y0 < point_y < y1:
        distances = {"left": abs(point_x - x0), "right": abs(x1 - point_x), "bottom": abs(point_y - y0), "top": abs(y1 - point_y)}
        side = min(distances, key=distances.get)
        if side == "left":
            nearest_x, nearest_y = x0, point_y
        elif side == "right":
            nearest_x, nearest_y = x1, point_y
        elif side == "bottom":
            nearest_x, nearest_y = point_x, y0
        else:
            nearest_x, nearest_y = point_x, y1
    return nearest_x, nearest_y


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

    Es la misma lógica de colocación que en el código sintético: calcula
    candidatos alrededor del punto, fuerza el recuadro hacia el interior cuando
    está cerca de un borde y añade candidatos de respaldo repartidos por el
    panel. Así se evitan solapes entre recuadros, estrellas y bordes.
    """
    box_width, box_height = estimate_label_box_size()

    min_tx = 0.35
    max_tx = max(0.35, max_x - box_width - 0.35)
    min_ty = box_height / 2 + 0.35
    max_ty = max(box_height / 2 + 0.35, max_y - box_height / 2 - 0.35)

    gap_x = clamp(0.070 * max_x, 1.25, 1.80)
    gap_y = clamp(0.055 * max_y, 1.05, 1.45)

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
    cortas o largas. Es la misma lógica usada en los códigos sintéticos/Real 1.
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


def draw_single_star_marker_2d(ax, x, y, color, size, angle_degrees=0.0, zorder=32):
    marker = (5, 1, angle_degrees)
    marker_size = max(7.0, np.sqrt(float(size)) * 1.12)
    ax.plot([x], [y], linestyle="None", marker=marker, markersize=marker_size,
            markerfacecolor=color, markeredgecolor=PIN_EDGE_COLOR, markeredgewidth=1.00, zorder=zorder)


def draw_2d_stars_with_overlap_control(ax, points):
    groups = {}
    for item in points:
        groups.setdefault((round(float(item["x"]), 2), round(float(item["y"]), 2)), []).append(item)
    for group_items in groups.values():
        n = len(group_items)
        if n == 1:
            item = group_items[0]
            draw_single_star_marker_2d(ax, item["x"], item["y"], item["color"], item["size"], 0.0, 32)
            continue
        angles = [-35.0, 0.0] if n == 2 else ([-40.0, 0.0, 40.0] if n == 3 else np.linspace(-42.0, 42.0, n).tolist())
        for idx, (item, angle) in enumerate(zip(group_items, angles)):
            draw_single_star_marker_2d(ax, item["x"], item["y"], item["color"], item["size"], angle, 32 + idx)


def nearest_point_on_text_bbox(ax, text_artist, point_x, point_y):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_display = text_artist.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
    inv = ax.transData.inverted()
    (x0, y0) = inv.transform((bbox_display.x0, bbox_display.y0))
    (x1, y1) = inv.transform((bbox_display.x1, bbox_display.y1))
    rect = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    return nearest_point_on_rect(point_x, point_y, rect)


def move_text_artist_inside_axes(ax, text_artist, pad_pixels=6):
    fig = ax.figure
    for _ in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_bbox = text_artist.get_window_extent(renderer=renderer)
        axes_bbox = ax.get_window_extent(renderer=renderer)
        dx_pixels = dy_pixels = 0.0
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
        new_x, new_y = ax.transData.inverted().transform((current_display[0] + dx_pixels, current_display[1] + dy_pixels))
        text_artist.set_position((new_x, new_y))
    return text_artist


def draw_contrast_reference_line(ax, x0, y0, x1, y1, color):
    length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if length < 0.12:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    marker_gap = min(0.34, 0.22 * length)
    box_gap = min(0.12, 0.10 * length)
    start_x, start_y = x0 + ux * marker_gap, y0 + uy * marker_gap
    end_x, end_y = x1 - ux * box_gap, y1 - uy * box_gap
    ax.plot([start_x, end_x], [start_y, end_y], color="white", linewidth=4.0, alpha=0.98, solid_capstyle="round", zorder=24, clip_on=False)
    ax.plot([start_x, end_x], [start_y, end_y], color=color, linewidth=1.65, alpha=0.98, solid_capstyle="round", zorder=25, clip_on=False)


def add_subtle_grid_points_2d(ax, pivot):
    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)
    x_pos = np.arange(len(c1_values))
    y_pos = np.arange(len(c0_values))
    X_grid, Y_grid = np.meshgrid(x_pos, y_pos)
    ax.scatter(X_grid.ravel(), Y_grid.ravel(), s=12, c="white", alpha=0.30, marker=".", edgecolors="none", zorder=8)


def add_2d_grid_note(fig):
    fig.text(0.50, 0.012, "Puntos blancos sutiles: combinaciones reales evaluadas de la rejilla manual; el fondo está suavizado/interpolado visualmente entre ellas.",
             ha="center", va="center", fontsize=8.0, color="#444444")


def add_pins_to_heatmap(ax, pivot, best_point, balanced_point, metric, scenario_name, val_comparison_df=None, equal_point=None):
    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)
    max_x = len(c1_values) - 1
    max_y = len(c0_values) - 1

    points = []
    for point, color, preferred in [
        (best_point, MODEL_COLORS["best_val_f1_grid"], "up"),
        (balanced_point, MODEL_COLORS["sklearn_balanced"], "down"),
        (equal_point, MODEL_COLORS["equal_weights"], "right"),
    ]:
        if point is None:
            continue
        points.append({
            "point": point,
            "color": color,
            "x": value_to_axis_position(point["c1"], c1_values),
            "y": value_to_axis_position(point["c0"], c0_values),
            "preferred": preferred,
            "size": 125,
        })

    points = choose_text_positions_for_all_pins(points, max_x, max_y)
    draw_2d_stars_with_overlap_control(ax, points)

    text_items = []
    for item in points:
        point = item["point"]
        metric_value = get_label_metric_value_for_pin(point, pivot, metric, scenario_name, val_comparison_df)
        text_artist = ax.text(
            item["text_x"], item["text_y"], format_pin_text(point, metric_value),
            fontsize=7.0, color=PIN_TEXT_COLOR, ha="left", va="center", zorder=30, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=PIN_EDGE_COLOR, alpha=0.94, linewidth=0.85),
        )
        text_items.append((item, text_artist))

    for _, text_artist in text_items:
        move_text_artist_inside_axes(ax, text_artist, pad_pixels=6)
    for item, text_artist in text_items:
        tx, ty = nearest_point_on_text_bbox(ax, text_artist, item["x"], item["y"])
        draw_contrast_reference_line(ax, item["x"], item["y"], tx, ty, item["color"])


def add_subtle_grid_points_3d(ax, X_grid, Y_grid, Z):
    ax.scatter(X_grid.ravel(), Y_grid.ravel(), Z.ravel(), s=14, c="white", alpha=0.30,
               marker=".", edgecolors="none", depthshade=False, zorder=230)


def draw_single_star_marker_3d(ax, x, y, z, color, angle_degrees=0.0, size=12.5, zorder=1000):
    marker = (5, 1, angle_degrees)
    ax.plot([x], [y], [z], linestyle="None", marker=marker, markersize=size,
            markerfacecolor=color, markeredgecolor=PIN_EDGE_COLOR, markeredgewidth=1.00, zorder=zorder)


def draw_3d_stars_with_overlap_control(ax, star_items):
    groups = {}
    for item in star_items:
        groups.setdefault((round(float(item["x"]), 2), round(float(item["y"]), 2)), []).append(item)
    for group_items in groups.values():
        n = len(group_items)
        if n == 1:
            item = group_items[0]
            draw_single_star_marker_3d(ax, item["x"], item["y"], item["z"], item["color"], 0.0, 12.5, 1000)
            continue
        angles = [-35.0, 0.0] if n == 2 else ([-40.0, 0.0, 40.0] if n == 3 else np.linspace(-42.0, 42.0, n).tolist())
        for idx, (item, angle) in enumerate(zip(group_items, angles)):
            draw_single_star_marker_3d(ax, item["x"], item["y"], item["z"], item["color"], angle, 12.5, 1000 + idx)


def add_pins_to_3d_surface(ax, pivot, best_point, balanced_point, metric, scenario_name, val_comparison_df=None, equal_point=None):
    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)
    pin_info, star_items = [], []

    for point, color, relation in [
        (best_point, MODEL_COLORS["best_val_f1_grid"], "punto de la rejilla manual"),
        (balanced_point, MODEL_COLORS["sklearn_balanced"], "referencia externa; no suele coincidir con un nodo de la rejilla"),
        (equal_point, MODEL_COLORS["equal_weights"], "estrategia final independiente; coincide con la rejilla si (1,1) está en C_VALUES"),
    ]:
        if point is None:
            continue
        x = value_to_axis_position(point["c1"], c1_values)
        y = value_to_axis_position(point["c0"], c0_values)
        z_surface = get_nearest_metric_value(pivot, point["c0"], point["c1"])
        z = z_surface if point["model_name"] == "best_val_f1_grid" else get_label_metric_value_for_pin(point, pivot, metric, scenario_name, val_comparison_df)
        star_items.append({"x": x, "y": y, "z": z, "color": color})
        pin_info.append({
            "label": point.get("short_label", point.get("label", "Punto")),
            "model_name": point.get("model_name"),
            "c0": point["c0"],
            "c1": point["c1"],
            "c0_std": float(point.get("c0_std", 0.0) or 0.0),
            "c1_std": float(point.get("c1_std", 0.0) or 0.0),
            "metric_value": z,
            "surface_reference": z_surface,
            "color": color,
            "marker": "*",
            "grid_relation": relation,
        })

    draw_3d_stars_with_overlap_control(ax, star_items)
    return pin_info



def add_3d_pin_legend(fig, metric, pin_info):
    handles = [
        Line2D([0], [0], marker=".", linestyle="None", markersize=12.5, markerfacecolor="#bfbfbf", markeredgecolor="#bfbfbf", markeredgewidth=0.0, alpha=0.75, label="Puntos: rejilla manual evaluada (c0, c1)"),
        Line2D([0], [0], color="#666666", linewidth=2.0, alpha=0.65, label="Superficie: interpolación visual"),
    ]
    for info in pin_info:
        weight_text = format_weights_for_pin_text(info)
        label = f"{info['label']}: {info['metric_value']:.3f} | {weight_text}"
        if info.get("model_name") == "sklearn_balanced":
            label += "\nref. externa a rejilla; class_weight='balanced'"
        handles.append(Line2D([0], [0], marker=info["marker"], linestyle="None", markersize=11.8,
                              markerfacecolor=info["color"], markeredgecolor=PIN_EDGE_COLOR,
                              markeredgewidth=1.05, label=label))
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.95, 0.055), fontsize=9.6,
               frameon=True, framealpha=0.96, ncol=1, borderpad=0.55, labelspacing=0.45,
               handlelength=1.5, handletextpad=0.6, title=f"Leyenda 3D ({metric_label(metric)})", title_fontsize=10.4)



def save_3d_surface_plot(agg_df, scenario_name, metric, filename, best_point=None, balanced_point=None, val_comparison_df=None, equal_point=None):
    temp_df = agg_df[agg_df["scenario"] == scenario_name].copy()
    if metric not in temp_df.columns or temp_df.empty:
        return
    pivot = temp_df.pivot(index="c0", columns="c1", values=metric)
    if pivot.empty or np.all(pd.isna(pivot.values)):
        return

    c0_values = pivot.index.to_numpy(dtype=float)
    c1_values = pivot.columns.to_numpy(dtype=float)
    x_pos = np.arange(len(c1_values))
    y_pos = np.arange(len(c0_values))
    X_grid, Y_grid = np.meshgrid(x_pos, y_pos)
    Z = pivot.values.astype(float)

    fig = plt.figure(figsize=(12.4, 8.9))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    surf = ax.plot_surface(X_grid, Y_grid, Z, cmap="viridis", edgecolor="none", alpha=0.65, antialiased=True, zorder=1)
    add_subtle_grid_points_3d(ax, X_grid, Y_grid, Z)

    max_pin_z = np.nanmax(Z)
    if best_point is not None and balanced_point is not None:
        pin_info = add_pins_to_3d_surface(ax, pivot, best_point, balanced_point, metric, scenario_name, val_comparison_df, equal_point=equal_point)
        add_3d_pin_legend(fig, metric, pin_info)
        for info in pin_info:
            max_pin_z = max(max_pin_z, info["metric_value"])

    ax.set_xlabel("c1", labelpad=32)
    ax.set_ylabel("c0", labelpad=16)
    ax.set_zlabel("")
    ax.set_title("")
    fig.suptitle(f"{pretty_scenario_label(scenario_name)} · Superficie 3D · {metric_label(metric)}", y=0.965, fontsize=13)
    ax.view_init(elev=31, azim=-56)

    step = 2
    ax.set_xticks(x_pos[::step])
    ax.set_yticks(y_pos[::step])
    ax.set_xticklabels([str(x) for x in c1_values[::step]], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(y) for y in c0_values[::step]], fontsize=8)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    ax.tick_params(axis="z", pad=6)

    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    z_range = z_max - z_min if z_max > z_min else 1.0
    ax.set_zlim(min(0, z_min - 0.04 * z_range), max_pin_z + 0.04 * z_range)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.72, aspect=18, pad=0.065)
    cbar.set_label(metric_label(metric), rotation=90, labelpad=12)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.tick_params(labelright=True, labelleft=False, right=True, left=False)
    plt.subplots_adjust(left=0.03, right=0.85, bottom=0.12, top=0.92)
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([cbar_pos.x0, cbar_pos.y0 + 0.07, cbar_pos.width, cbar_pos.height])
    fig.savefig(get_scenario_figure_path(scenario_name, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_panel(agg_df, scenario_name, metrics_panel, title, filename, best_point=None, balanced_point=None, val_comparison_df=None, equal_point=None):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    axes = axes.ravel()
    temp_df = agg_df[agg_df["scenario"] == scenario_name].copy()

    for ax, metric in zip(axes, metrics_panel):
        if metric not in temp_df.columns:
            ax.axis("off")
            continue
        pivot = temp_df.pivot(index="c0", columns="c1", values=metric)
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", interpolation="hanning")
        add_subtle_grid_points_2d(ax, pivot)
        if best_point is not None and balanced_point is not None:
            add_pins_to_heatmap(ax, pivot, best_point, balanced_point, metric, scenario_name, val_comparison_df, equal_point=equal_point)
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
        ax.set_title(metric_label(metric, include_split=False), fontsize=11)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(metric_label(metric), rotation=90, labelpad=12)

    fig.suptitle(title, fontsize=15)
    add_2d_grid_note(fig)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(get_scenario_figure_path(scenario_name, filename), dpi=180)
    plt.close(fig)



def save_validation_figures(agg_df, best_df, balanced_df, val_comparison_df=None):
    validation_mean_panels = [
        {"name": "val_main_mean", "title": "VALIDACIÓN · Mean · Métricas principales", "metrics": ["val_f1_mean", "val_pr_auc_mean", "val_balanced_accuracy_mean", "val_mcc_mean"]},
        {"name": "val_metrics_per_class_mean", "title": "VALIDACIÓN · Mean · Métricas por clase", "metrics": ["val_recall_mean", "val_fnr_mean", "val_specificity_mean", "val_roc_auc_mean"]},
    ]
    validation_std_panels = [
        {"name": "val_main_std", "title": "VALIDACIÓN · STD · Métricas principales", "metrics": ["val_f1_std", "val_pr_auc_std", "val_balanced_accuracy_std", "val_mcc_std"]},
        {"name": "val_metrics_per_class_std", "title": "VALIDACIÓN · STD · Métricas por clase", "metrics": ["val_recall_std", "val_fnr_std", "val_specificity_std", "val_roc_auc_std"]},
    ]
    for scenario_name in SCENARIOS:
        best_point, balanced_point, equal_point = get_pin_points(best_df, balanced_df, scenario_name)
        scen_label = pretty_scenario_label(scenario_name)

        for panel in validation_mean_panels:
            save_heatmap_panel(agg_df, scenario_name, panel["metrics"], f"{scen_label} · {panel['title']} ({len(SEEDS)} seeds)", f"{scenario_short(scenario_name)}_panel_{panel['name']}_with_pins.png", best_point, balanced_point, val_comparison_df, equal_point=equal_point)
            for metric in panel["metrics"]:
                save_3d_surface_plot(agg_df, scenario_name, metric, f"{scenario_short(scenario_name)}_3d_{metric}_with_pins.png", best_point, balanced_point, val_comparison_df, equal_point=equal_point)

        for panel in validation_std_panels:
            save_heatmap_panel(agg_df, scenario_name, panel["metrics"], f"{scen_label} · {panel['title']} ({len(SEEDS)} seeds)", f"{scenario_short(scenario_name)}_panel_{panel['name']}.png")
            for metric in panel["metrics"]:
                save_3d_surface_plot(agg_df, scenario_name, metric, f"{scenario_short(scenario_name)}_3d_{metric}.png")



def save_final_bar_plots(agg_df, split_name):
    metrics = [
        f"{split_name}_f1_mean", f"{split_name}_pr_auc_mean", f"{split_name}_balanced_accuracy_mean", f"{split_name}_mcc_mean",
        f"{split_name}_recall_mean", f"{split_name}_fnr_mean", f"{split_name}_specificity_mean", f"{split_name}_roc_auc_mean",
    ]
    model_order = [m for m in FINAL_MODEL_NAMES if m in set(agg_df["model_name"].unique())]
    scenario_order = [s for s in SCENARIOS if s in set(agg_df["scenario"].unique())]

    for metric in metrics:
        if metric not in agg_df.columns:
            continue
        std_col = metric.replace("_mean", "_std")
        x = np.arange(len(scenario_order))
        width = 0.24 if len(model_order) >= 3 else 0.32
        fig, ax = plt.subplots(figsize=(14.2, 7.0))
        all_values = []
        for m_idx, model_name in enumerate(model_order):
            style = get_model_plot_style(model_name)
            values, errors, c0_values, c1_values = [], [], [], []
            for scenario_name in scenario_order:
                row = agg_df[(agg_df["scenario"] == scenario_name) & (agg_df["model_name"] == model_name)]
                if row.empty:
                    values.append(np.nan); errors.append(0.0); c0_values.append(np.nan); c1_values.append(np.nan)
                    continue
                row = row.iloc[0]
                values.append(float(row[metric]))
                errors.append(float(row[std_col]) if std_col in agg_df.columns and not pd.isna(row[std_col]) else 0.0)
                c0_values.append(float(row.get("c0_mean", np.nan)))
                c1_values.append(float(row.get("c1_mean", np.nan)))
            values = np.asarray(values, dtype=float)
            errors = np.asarray(errors, dtype=float)
            all_values.extend(values[~np.isnan(values)].tolist())
            offset = (m_idx - (len(model_order) - 1) / 2.0) * width
            bars = ax.bar(x + offset, values, width=width, label=style["label"], color=style["color"], edgecolor="white", linewidth=0.8,
                          yerr=errors, error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0, "ecolor": "#333333", "alpha": 0.85})
            for bar, value, error, c0, c1 in zip(bars, values, errors, c0_values, c1_values):
                if pd.isna(value):
                    continue
                text = f"mean={value:.3f}\nstd={error:.3f}\nc0={format_weight_for_bar(c0)}\nc1={format_weight_for_bar(c1)}"
                if value >= 0.28:
                    y, va, color, bbox = value - 0.035, "top", "white", dict(boxstyle="round,pad=0.16", fc="black", ec="none", alpha=0.24)
                else:
                    y, va, color, bbox = value + max(0.035, error + 0.020), "bottom", "#222222", dict(boxstyle="round,pad=0.16", fc="white", ec=style["color"], alpha=0.92, linewidth=0.6)
                ax.text(bar.get_x() + bar.get_width()/2, y, text, ha="center", va=va, fontsize=7.1, color=color, bbox=bbox, zorder=30)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_scenario_label(s) for s in scenario_order], rotation=0, ha="center", fontsize=10)
        ax.set_xlabel("Escenario", labelpad=10)
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"Comparación final en {format_split_label_for_plot(split_name)} · {metric_label(metric, include_split=False)}", fontsize=14, pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        for boundary in np.arange(0.5, len(scenario_order) - 0.5, 1.0):
            ax.axvline(boundary, color="#dddddd", linewidth=0.8, linestyle="--", alpha=0.7)
        if all_values:
            min_v, max_v = min(all_values), max(all_values)
            lower, upper = min(0.0, min_v - 0.10), max(1.0, max_v + 0.20)
            if metric.endswith("_mcc_mean"):
                lower, upper = min(-0.05, min_v - 0.12), max(0.20, max_v + 0.20)
            ax.set_ylim(lower, upper)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(model_order), frameon=True, fontsize=9)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(get_global_figure_path(f"{split_name}_final_comparison_{metric}.png"), dpi=180)
        plt.close(fig)


def save_boxplots(raw_df, split_name):
    metrics = [
        f"{split_name}_f1", f"{split_name}_pr_auc", f"{split_name}_balanced_accuracy", f"{split_name}_mcc",
        f"{split_name}_recall", f"{split_name}_fnr", f"{split_name}_specificity", f"{split_name}_roc_auc",
    ]
    model_order = [m for m in FINAL_MODEL_NAMES if m in set(raw_df["model_name"].unique())]
    scenario_order = [s for s in SCENARIOS if s in set(raw_df["scenario"].unique())]
    summary_rows = []

    for metric in metrics:
        if metric not in raw_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(14.8, 7.8))
        box_data, box_positions, box_colors = [], [], []
        group_width = len(model_order) + 1
        xtick_positions, xtick_labels = [], []
        for s_idx, scenario_name in enumerate(scenario_order):
            base = s_idx * group_width
            xtick_positions.append(base + (len(model_order) - 1) / 2.0)
            xtick_labels.append(pretty_scenario_label(scenario_name))
            for m_idx, model_name in enumerate(model_order):
                values = raw_df[(raw_df["scenario"] == scenario_name) & (raw_df["model_name"] == model_name)][metric].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                style = get_model_plot_style(model_name)
                box_data.append(values)
                box_positions.append(base + m_idx)
                box_colors.append(style["color"])
                q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
                summary_rows.append({
                    "split": split_name, "metric": metric, "scenario": scenario_name, "model_name": model_name,
                    "n_seeds": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)), "q1": q1, "q3": q3, "iqr": q3 - q1,
                    "min": float(np.min(values)), "max": float(np.max(values)),
                })
        if not box_data:
            plt.close(fig)
            continue
        boxplot = ax.boxplot(box_data, positions=box_positions, widths=0.58, patch_artist=True, showmeans=True,
                             meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=1.05, markersize=5.2),
                             flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=0.9, markersize=3.8, linestyle="none", alpha=0.95))
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
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=10)
        ax.set_xlabel("Escenario", labelpad=10)
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"Caja y bigotes en {format_split_label_for_plot(split_name)} · {metric_label(metric, include_split=False)}", fontsize=14, pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        for scenario_idx in range(1, len(scenario_order)):
            ax.axvline(scenario_idx * group_width - 0.5, color="#dddddd", linewidth=0.8, linestyle="--", alpha=0.7)
        model_handles = [Line2D([0], [0], color=get_model_plot_style(m)["color"], linewidth=6.0, solid_capstyle="butt", label=get_model_plot_style(m)["label"]) for m in model_order]
        reading_handles = [
            Line2D([0], [0], marker="s", linestyle="None", markersize=8, markerfacecolor="white", markeredgecolor="#222222", label="Caja: Q1-Q3 (50% central)"),
            Line2D([0], [0], color="#111111", linewidth=1.4, label="Línea negra: mediana"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=5.2, markerfacecolor="white", markeredgecolor="#111111", label="Círculo blanco: media"),
            Line2D([0], [0], color="#333333", linewidth=1.0, label="Bigotes: rango no atípico"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=3.8, markerfacecolor="white", markeredgecolor="#111111", label="Puntos fuera: valores atípicos"),
        ]
        fig.legend(handles=reading_handles, loc="center right", bbox_to_anchor=(0.985, 0.56), frameon=True, framealpha=0.96, fontsize=8.4, title="Lectura caja-bigotes", title_fontsize=9.2)
        fig.legend(handles=model_handles, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=len(model_order), frameon=True, fontsize=9.5)
        fig.tight_layout(rect=[0, 0.16, 0.80, 1])
        fig.savefig(get_global_figure_path(f"{split_name}_final_boxplot_{metric}.png"), dpi=180, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / f"{split_name}_final_boxplot_summary_table.csv", index=False)
    return summary


def _draw_confusion_matrix_on_axis(ax, cm, title):
    cm = np.asarray(cm, dtype=int)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
    ax.set_yticklabels(["Real 0", "Real 1"], fontsize=9)
    ax.set_xlabel("Predicción", fontsize=9)
    ax.set_ylabel("Clase real", fontsize=9)
    ax.set_title(title, fontsize=10.0, pad=12)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] >= 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j] * 100:.1f}%", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def save_confusion_matrices(pred_df, split_name):
    rows = []
    model_order = [m for m in FINAL_MODEL_NAMES if m in set(pred_df["model_name"].unique())]
    scenario_order = [s for s in SCENARIOS if s in set(pred_df["scenario"].unique())]

    for model_name in model_order:
        style = get_model_plot_style(model_name)
        model_df = pred_df[pred_df["model_name"] == model_name].copy()
        ncols = min(2, max(1, len(scenario_order)))
        nrows = int(np.ceil(len(scenario_order) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.8 * ncols, 7.0 * nrows))
        axes = np.asarray(axes).ravel()
        last_im = None
        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(scenario_order):
                ax.axis("off")
                continue
            scenario_name = scenario_order[ax_idx]
            group = model_df[model_df["scenario"] == scenario_name].copy()
            if group.empty:
                ax.axis("off")
                continue
            y_true = group["y_true"].to_numpy(dtype=int)
            y_pred = group["y_pred"].to_numpy(dtype=int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            rows.append({"split": split_name, "scenario": scenario_name, "model_name": model_name, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
            cm = np.array([[tn, fp], [fn, tp]], dtype=int)
            n_total, n_pos = int(len(group)), int(np.sum(y_true == 1))
            n_seeds_group = int(group["seed"].nunique()) if "seed" in group.columns else 0
            title = f"{pretty_scenario_label(scenario_name)}\n{format_split_label_for_plot(split_name)} · agregada {n_seeds_group} seeds\nn total={n_total}, positivos={n_pos}"
            last_im = _draw_confusion_matrix_on_axis(ax, cm, title)

        n_seeds_panel = int(model_df["seed"].nunique()) if "seed" in model_df.columns else len(SEEDS)
        fig.suptitle(f"{format_split_label_for_plot(split_name)} FINAL · Matrices de confusión agregadas entre {n_seeds_panel} seeds · {style['label']}", fontsize=16.5, y=0.972)
        fig.text(0.5, 0.045, f"Lectura: cada matriz acumula las predicciones finales de {format_split_label_for_plot(split_name)} obtenidas en {n_seeds_panel} seeds. Cada celda muestra conteo acumulado y porcentaje por fila real.", ha="center", va="center", fontsize=8.7, color="#444444", wrap=True)
        fig.subplots_adjust(left=0.060, right=0.895, bottom=0.135, top=0.790, wspace=0.18, hspace=0.42)
        if last_im is not None:
            cbar_ax = fig.add_axes([0.930, 0.235, 0.018, 0.470])
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label("Porcentaje dentro de cada clase real", rotation=90, labelpad=12)
        fig.savefig(get_global_figure_path(f"{split_name}_confusion_matrix_panel_{model_name}.png"), dpi=180, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)

    cm_df = pd.DataFrame(rows)
    cm_df.to_csv(OUTPUT_DIR / f"{split_name}_final_confusion_matrices.csv", index=False)
    return cm_df

# ============================================================
# 7. TABLAS Y TIEMPOS
# ============================================================

def print_comparison_table_by_scenario(df, title, split_name, decimals=4):
    df_print = df.copy().round(decimals)
    blocks = [
        ("Pesos de clase", ["scenario", "target", "model_name", "c0_mean", "c0_std", "c1_mean", "c1_std"]),
        ("Métricas principales", ["scenario", "target", "model_name", f"{split_name}_f1_mean", f"{split_name}_f1_std", f"{split_name}_pr_auc_mean", f"{split_name}_pr_auc_std", f"{split_name}_balanced_accuracy_mean", f"{split_name}_balanced_accuracy_std", f"{split_name}_mcc_mean", f"{split_name}_mcc_std"]),
        ("Métricas por clase y curvas", ["scenario", "target", "model_name", f"{split_name}_recall_mean", f"{split_name}_recall_std", f"{split_name}_fnr_mean", f"{split_name}_fnr_std", f"{split_name}_specificity_mean", f"{split_name}_specificity_std", f"{split_name}_roc_auc_mean", f"{split_name}_roc_auc_std"]),
    ]

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print("Nota: se muestran media y desviación típica entre las 20 semillas/particiones estratificadas.")

    for scenario_name in SCENARIOS:
        temp = df_print[df_print["scenario"] == scenario_name].copy()
        if temp.empty:
            continue
        print("\n" + "#" * 110)
        print(f"ESCENARIO: {scenario_name.upper()}")
        print("#" * 110)
        for block_title, cols in blocks:
            existing = [c for c in cols if c in temp.columns]
            if len(existing) <= 3:
                continue
            print("\n" + "-" * 110)
            print(block_title)
            print("-" * 110)
            with pd.option_context("display.max_columns", None, "display.width", 240):
                print(temp[existing].to_string(index=False))


def save_timers(timer_frames):
    frames = [df for df in timer_frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    timers = pd.concat(frames, ignore_index=True)

    useful_stages = [
        "load_existing_grid_coefficients",
        "grid_train_scenario_parallel_total",
        "grid_val_evaluation_scenario_parallel_total",
        "grid_aggregation_by_scenario",
        "grid_best_selection_by_scenario",
        "load_existing_final_equal_weights_coefficients",
        "load_existing_final_sklearn_balanced_coefficients",
        "final_equal_weights_train_scenario_parallel_total",
        "final_sklearn_balanced_train_scenario_parallel_total",
        "final_best_val_f1_grid_val_evaluation_scenario_parallel_total",
        "final_equal_weights_val_evaluation_scenario_parallel_total",
        "final_sklearn_balanced_val_evaluation_scenario_parallel_total",
        "final_best_val_f1_grid_test_evaluation_scenario_parallel_total",
        "final_equal_weights_test_evaluation_scenario_parallel_total",
        "final_sklearn_balanced_test_evaluation_scenario_parallel_total",
        "figures_total",
        "script_total",
    ]

    timers = timers[
        (timers["model_name"].astype(str) == "wall_clock")
        & (timers["stage"].astype(str).isin(useful_stages))
    ].copy()

    cols = ["scenario", "target", "seed", "stage", "model_name", "n_models", "total_seconds", "total_time_readable", "skipped_because_existing_outputs"]
    return timers[[c for c in cols if c in timers.columns]].copy()


def print_clean_timer_summary(timers):
    if timers is None or timers.empty:
        return

    timers = timers.copy()
    timers["total_seconds"] = pd.to_numeric(timers["total_seconds"], errors="coerce")
    timers["n_models"] = pd.to_numeric(timers["n_models"], errors="coerce")

    grid_size = get_grid_models_per_seed()
    n_seeds = len(SEEDS)
    grid_train_models_total = get_total_grid_train_models()
    single_strategy_train_models_total = get_total_single_strategy_train_models()

    def seconds(stage, scenario=None):
        rows = timers[timers["stage"].astype(str) == stage]
        if scenario is not None:
            rows = rows[rows["scenario"].astype(str) == str(scenario)]
        if rows.empty:
            return np.nan
        return float(rows["total_seconds"].sum())

    def nmodels(stage, scenario=None, fallback=None):
        rows = timers[timers["stage"].astype(str) == stage]
        if scenario is not None:
            rows = rows[rows["scenario"].astype(str) == str(scenario)]

        if rows.empty or "n_models" not in rows.columns:
            if fallback is not None and not pd.isna(fallback):
                return int(fallback)
            return 0

        value = rows["n_models"].fillna(0.0).sum()

        if value == 0 and fallback is not None and not pd.isna(fallback):
            return int(fallback)

        return int(value)

    def fmt(value):
        return "-" if pd.isna(value) else format_seconds(value)

    def safe_sum(values):
        total = 0.0
        any_value = False
        for value in values:
            if not pd.isna(value):
                total += float(value)
                any_value = True
        return total if any_value else np.nan

    print("\n" + "=" * 128)
    print("RESUMEN DE COSTE COMPUTACIONAL COMPARABLE · DATASET REAL 2 · PARTICIÓN CLÁSICA")
    print("=" * 128)
    print("Lectura rápida:")
    print("- Solo se muestran tiempos reales / wall-clock.")
    print("- Se usa split estratificado TRAIN/VAL/TEST con 20 semillas, igual que en los otros códigos.")
    print("- TRAIN rejilla: entrena todos los pesos sobre TRAIN y guarda coeficientes wide en grid_manual.")
    print("- VAL rejilla: carga coeficientes y evalúa VAL para todos los pares (c0, c1), sin reentrenar.")
    print("- Agregación/selección: calcula mean/std entre seeds y elige Best F1 por val_f1_mean.")
    print("- TEST final: carga coeficientes guardados y evalúa TEST, sin volver a entrenar ni seleccionar.")
    print("- No existe carpeta final_models: Best F1 sale de grid_manual; Equal y Sklearn se guardan en sus carpetas independientes.")
    print(f"- Rejilla actual: {len(C_VALUES)} x {len(C_VALUES)} = {grid_size} combinaciones por seed x {n_seeds} seeds.")
    print("- Risk 3 y Risk 4 se excluyen por no tener suficientes positivos para partición clásica fiable.")

    print("\n0.0) COSTE COMPUTACIONAL COMPARABLE HASTA TEST")
    print("   Tabla principal: TRAIN + VAL rejilla + agregación/selección + TEST final.")

    for scenario_name in SCENARIOS:
        train_grid = seconds("grid_train_scenario_parallel_total", scenario_name)
        val_grid = seconds("grid_val_evaluation_scenario_parallel_total", scenario_name)
        agg_s = seconds("grid_aggregation_by_scenario", scenario_name)
        sel_s = seconds("grid_best_selection_by_scenario", scenario_name)
        agg_sel = safe_sum([agg_s, sel_s])
        best_test = seconds("final_best_val_f1_grid_test_evaluation_scenario_parallel_total", scenario_name)
        best_total = safe_sum([train_grid, val_grid, agg_s, sel_s, best_test])

        rows = [{
            "estrategia": "Manual Best F1",
            "modelos entrenados TRAIN": nmodels(
                "grid_train_scenario_parallel_total",
                scenario_name,
                fallback=grid_train_models_total,
            ),
            "TRAIN real": fmt(train_grid),
            "Validación rejilla": fmt(val_grid),
            "Agregación/selección": fmt(agg_sel),
            "TEST final": fmt(best_test),
            "Total comparable": fmt(best_total),
        }]

        for model_name in ["equal_weights", "sklearn_balanced"]:
            train_s = seconds(f"final_{model_name}_train_scenario_parallel_total", scenario_name)
            test_s = seconds(f"final_{model_name}_test_evaluation_scenario_parallel_total", scenario_name)
            total = safe_sum([train_s, test_s])
            rows.append({
                "estrategia": MODEL_LABELS[model_name],
                "modelos entrenados TRAIN": nmodels(
                    f"final_{model_name}_train_scenario_parallel_total",
                    scenario_name,
                    fallback=single_strategy_train_models_total,
                ),
                "TRAIN real": fmt(train_s),
                "Validación rejilla": "No aplica",
                "Agregación/selección": "No aplica",
                "TEST final": fmt(test_s),
                "Total comparable": fmt(total),
            })

        print("\n" + "#" * 118)
        print(f"ESCENARIO: {scenario_name.upper()}")
        print("#" * 118)
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n0.1) DESGLOSE DEL ENTRENAMIENTO")
    train_rows = []
    for scenario_name in SCENARIOS:
        train_rows.append({
            "escenario": scenario_name,
            "TRAIN rejilla": fmt(seconds("grid_train_scenario_parallel_total", scenario_name)),
            "modelos rejilla": nmodels(
                "grid_train_scenario_parallel_total",
                scenario_name,
                fallback=grid_train_models_total,
            ),
            "TRAIN Equal": fmt(seconds("final_equal_weights_train_scenario_parallel_total", scenario_name)),
            "modelos Equal": nmodels(
                "final_equal_weights_train_scenario_parallel_total",
                scenario_name,
                fallback=single_strategy_train_models_total,
            ),
            "TRAIN Sklearn": fmt(seconds("final_sklearn_balanced_train_scenario_parallel_total", scenario_name)),
            "modelos Sklearn": nmodels(
                "final_sklearn_balanced_train_scenario_parallel_total",
                scenario_name,
                fallback=single_strategy_train_models_total,
            ),
            "Total TRAIN": fmt(safe_sum([
                seconds("grid_train_scenario_parallel_total", scenario_name),
                seconds("final_equal_weights_train_scenario_parallel_total", scenario_name),
                seconds("final_sklearn_balanced_train_scenario_parallel_total", scenario_name),
            ])),
        })
    print("   Weighted grid entrena todos los pares (c0, c1) de la rejilla y guarda coeficientes.")
    print("   Equal weights y Sklearn balanced entrenan un único modelo por seed.")
    print(pd.DataFrame(train_rows).to_string(index=False))

    print("\n0.2) VALIDACIÓN DE LA REJILLA")
    val_rows = []
    for scenario_name in SCENARIOS:
        val_rows.append({
            "escenario": scenario_name,
            "VAL rejilla": fmt(seconds("grid_val_evaluation_scenario_parallel_total", scenario_name)),
            "puntos evaluados": nmodels(
                "grid_val_evaluation_scenario_parallel_total",
                scenario_name,
                fallback=grid_train_models_total,
            ),
        })
    print("   No se reentrena: se cargan los coeficientes entrenados en TRAIN y se evalúan todos los pesos en VALIDACIÓN.")
    print(pd.DataFrame(val_rows).to_string(index=False))

    print("\n0.3) AGREGACIÓN Y SELECCIÓN BEST F1")
    agg_rows = []
    for scenario_name in SCENARIOS:
        agg_s = seconds("grid_aggregation_by_scenario", scenario_name)
        sel_s = seconds("grid_best_selection_by_scenario", scenario_name)
        n_points = nmodels("grid_best_selection_by_scenario", scenario_name)
        agg_rows.append({
            "escenario": scenario_name,
            "agregación mean/std": fmt(agg_s),
            "selección Best F1": fmt(sel_s),
            "total agregación/selección": fmt(safe_sum([agg_s, sel_s])),
            "puntos revisados": n_points if n_points > 0 else grid_size,
            "criterio": "max val_f1_mean",
        })
    print(f"   Agrega mean/std entre las {n_seeds} seeds por escenario y selecciona el mayor val_f1_mean.")
    print("   Este tiempo se incluye en el coste comparable de Manual Best F1.")
    print(pd.DataFrame(agg_rows).to_string(index=False))

    print("\n0.4) TEST FINAL")
    test_rows = []
    for scenario_name in SCENARIOS:
        test_rows.append({
            "escenario": scenario_name,
            "TEST Best F1": fmt(seconds("final_best_val_f1_grid_test_evaluation_scenario_parallel_total", scenario_name)),
            "TEST Equal": fmt(seconds("final_equal_weights_test_evaluation_scenario_parallel_total", scenario_name)),
            "TEST Sklearn": fmt(seconds("final_sklearn_balanced_test_evaluation_scenario_parallel_total", scenario_name)),
            "Total TEST": fmt(safe_sum([
                seconds("final_best_val_f1_grid_test_evaluation_scenario_parallel_total", scenario_name),
                seconds("final_equal_weights_test_evaluation_scenario_parallel_total", scenario_name),
                seconds("final_sklearn_balanced_test_evaluation_scenario_parallel_total", scenario_name),
            ])),
        })
    print("   En TEST no se selecciona nada y no se reentrena: se cargan coeficientes guardados y se evalúan las estrategias finales.")
    print(pd.DataFrame(test_rows).to_string(index=False))

    print("\n0.5) TOTAL REAL POR ESCENARIO")
    total_rows = []
    for scenario_name in SCENARIOS:
        weighted_total = safe_sum([
            seconds("grid_train_scenario_parallel_total", scenario_name),
            seconds("grid_val_evaluation_scenario_parallel_total", scenario_name),
            seconds("grid_aggregation_by_scenario", scenario_name),
            seconds("grid_best_selection_by_scenario", scenario_name),
            seconds("final_best_val_f1_grid_test_evaluation_scenario_parallel_total", scenario_name),
        ])
        equal_total = safe_sum([
            seconds("final_equal_weights_train_scenario_parallel_total", scenario_name),
            seconds("final_equal_weights_test_evaluation_scenario_parallel_total", scenario_name),
        ])
        sklearn_total = safe_sum([
            seconds("final_sklearn_balanced_train_scenario_parallel_total", scenario_name),
            seconds("final_sklearn_balanced_test_evaluation_scenario_parallel_total", scenario_name),
        ])
        total_rows.append({
            "escenario": scenario_name,
            "Weighted Best F1": fmt(weighted_total),
            "Equal weights": fmt(equal_total),
            "Sklearn balanced": fmt(sklearn_total),
            "Bloque completo comparable": fmt(safe_sum([weighted_total, equal_total, sklearn_total])),
        })
    print("   Weighted Best F1 = TRAIN rejilla + VAL rejilla + agregación/selección + TEST Best F1.")
    print("   Equal weights = TRAIN Equal + TEST Equal. Sklearn balanced = TRAIN Sklearn + TEST Sklearn.")
    print(pd.DataFrame(total_rows).to_string(index=False))

    other = []
    for stage, label in [("figures_total", "Generación de figuras"), ("script_total", "Script completo")]:
        value = seconds(stage, "all")
        if not pd.isna(value):
            other.append({"bloque": label, "tiempo real": format_seconds(value)})
    if other:
        print("\n0.6) OTROS TIEMPOS REALES")
        print("   Estos tiempos se guardan aparte porque no forman parte del coste experimental comparable hasta TEST.")
        print(pd.DataFrame(other).to_string(index=False))

    print("\nResumen de tiempos comparables guardado en timers_execution.txt.")


def write_experiment_metadata():
    metadata = {
        "run_signature": RUN_SIGNATURE,
        "model": "LogisticRegression estándar de Scikit-Learn",
        "logistic_regression_config": {
            "class_weight": "variable según c0/c1 o balanced",
            "default_solver": "lbfgs",
            "default_penalty": "l2",
            "max_iter": MODEL_MAX_ITER,
        },
        "validation_strategy": "classic_train_val_test_stratified",
        "split": {"train": TRAIN_SIZE_FINAL, "val": VAL_SIZE_FINAL, "test": TEST_SIZE_FINAL},
        "execution_structure_terminal": [
            "TRAIN: entrenar con la partición TRAIN y guardar coeficientes",
            "VALIDACIÓN: cargar coeficientes y evaluar VAL sin reentrenar",
            "SELECCIÓN: elegir el mejor punto de pesos según val_f1_mean",
            "TEST: cargar coeficientes finales guardados y evaluar TEST sin reentrenar",
        ],
        "risk_targets_note": "Risk 3 y Risk 4 se eliminan como targets activos por tener muy pocos positivos para partición clásica fiable.",
        "c_values": C_VALUES,
        "grid_size": len(C_VALUES) * len(C_VALUES),
        "model_max_iter": MODEL_MAX_ITER,
        "targets": TARGET_COLS,
        "seeds": SEEDS,
        "dataset": DATA_FILE_NAME,
        "features_type": "quantitative_only",
        "missing_policy": "drop_rows_with_missing_in_selected_quantitative_features_or_target",
        "imputation": False,
        "coefficient_structure": {
            "grid_manual": str(GRID_MANUAL_COEFFICIENTS_WIDE_DIR.resolve()),
            "equal_weights": str(EQUAL_WEIGHTS_COEFFICIENTS_WIDE_DIR.resolve()),
            "sklearn_balanced": str(SKLEARN_BALANCED_COEFFICIENTS_WIDE_DIR.resolve()),
            "best_val_f1_grid": "uses selected row from grid_manual coefficients; no duplicated final_models folder",
        },
        "output_dir": str(OUTPUT_DIR.resolve()),
    }
    with open(PATHS["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


# ============================================================
# 8. BUCLE PRINCIPAL
# ============================================================

def main():
    total_start = time.perf_counter()
    timer_frames = []

    print("PROJECT_DIR =", PROJECT_DIR.resolve())
    print("OUTPUT_DIR =", OUTPUT_DIR.resolve())
    print("DATA_FILE_PATH =", DATA_FILE_PATH.resolve(), DATA_FILE_PATH.exists())
    print("Validación = partición clásica estratificada TRAIN/VAL/TEST")
    print(f"Split = {TRAIN_SIZE_FINAL:.0%}/{VAL_SIZE_FINAL:.0%}/{TEST_SIZE_FINAL:.0%}")
    print(f"Semillas = {len(SEEDS)}")
    print("Estructura:")
    print("  1) TRAIN: entrenar rejilla y guardar coeficientes.")
    print("  2) VALIDACIÓN: cargar coeficientes y evaluar VAL.")
    print("  3) SELECCIÓN: Best F1 por val_f1_mean.")
    print("  4) TEST: cargar coeficientes finales y evaluar TEST.")
    print("No hay Leave-One-Out en este código.")

    _, feature_cols, clean_df_by_scenario, _ = load_and_clean_real_dataset(print_terminal_summary=True)
    build_all_seed_datasets(clean_df_by_scenario, feature_cols)
    run_parallel_warmup(clean_df_by_scenario, feature_cols)
    write_experiment_metadata()

    pd.DataFrame([
        {
            "scenario": name,
            "dataset": DATA_FILE_NAME,
            "target": cfg["target"],
            "features_type": "quantitative_only",
            "validation_strategy": "classic_train_val_test_stratified",
            "train_size": TRAIN_SIZE_FINAL,
            "val_size": VAL_SIZE_FINAL,
            "test_size": TEST_SIZE_FINAL,
            "n_seeds": len(SEEDS),
            "n_c_values": len(C_VALUES),
            "grid_size": len(C_VALUES) * len(C_VALUES),
            "missing_policy": "drop_rows_with_missing_in_selected_quantitative_features_or_target",
            "imputation": False,
            "model_max_iter": MODEL_MAX_ITER,
            "solver": "lbfgs_default",
            "excluded_columns": f"{ID_COL} and all Suicide_Risk targets",
        }
        for name, cfg in SCENARIOS.items()
    ]).to_csv(PATHS["scenarios_config"], index=False)

    print("\n" + "#" * 100)
    print("FASE 1 · TRAIN REJILLA / FASE 2 · VALIDACIÓN / FASE 3 · SELECCIÓN")
    print("#" * 100)

    _, _, grid_agg, best_df, balanced_df = evaluate_grid_validation(clean_df_by_scenario, feature_cols, timer_frames)

    print("\nMejor configuración según F1 medio en VALIDACIÓN:")
    best_cols = ["scenario", "target", "c0", "c1", "val_f1_mean", "val_pr_auc_mean", "val_balanced_accuracy_mean", "val_mcc_mean", "val_recall_mean", "val_fnr_mean", "val_specificity_mean", "val_roc_auc_mean"]
    print(best_df[[c for c in best_cols if c in best_df.columns]].to_string(index=False))

    print("\nPuntos equivalentes a class_weight='balanced' de Scikit-Learn:")
    print(balanced_df.to_string(index=False))

    print("\n" + "#" * 100)
    print("FASE 4 · COMPARACIÓN FINAL VAL Y TEST")
    print("#" * 100)

    final_outputs = {}
    for split_name in ["val", "test"]:
        raw, pred, agg, comp = evaluate_final_split(clean_df_by_scenario, feature_cols, best_df, split_name, timer_frames)
        final_outputs[split_name] = {"raw": raw, "pred": pred, "agg": agg, "comp": comp}
        print_comparison_table_by_scenario(comp, f"Tabla comparativa final · {split_name.upper()}", split_name, decimals=4)

    print("\nResultados principales guardados en:")
    for key in ["val_grid_raw", "val_grid_agg", "best_configs", "balanced", "val_final_comp", "test_final_comp"]:
        print("-", PATHS[key])

    start_fig = time.perf_counter()
    save_validation_figures(grid_agg, best_df, balanced_df, final_outputs["val"]["comp"])
    for split_name in ["val", "test"]:
        save_final_bar_plots(final_outputs[split_name]["agg"], split_name)
        save_boxplots(final_outputs[split_name]["raw"], split_name)
        save_confusion_matrices(final_outputs[split_name]["pred"], split_name)
    add_wall_clock_timer(timer_frames, "all", "figures_total", start_fig, n_models=0, skipped=False)

    add_wall_clock_timer(timer_frames, "all", "script_total", total_start, n_models=np.nan, skipped=False)

    timers = save_timers(timer_frames)

    import io
    from contextlib import redirect_stdout
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_clean_timer_summary(timers)
    text = buffer.getvalue()
    print(text, end="")
    with open(PATHS["timers"], "w", encoding="utf-8") as f:
        f.write(text)

    write_experiment_metadata()

    print("\nResumen de tiempos guardado en:")
    print(PATHS["timers"])
    print("\nTodo terminado correctamente.")
    print("Resultados, datasets, predicciones, comparación final, coeficientes, timers y figuras guardados en:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
