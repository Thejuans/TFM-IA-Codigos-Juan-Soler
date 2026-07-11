# TFM - Clasificación binaria desbalanceada mediante ponderación de clases

Este repositorio contiene los códigos desarrollados para el Trabajo Fin de Máster centrado en el análisis de problemas de clasificación binaria desbalanceada mediante regresión logística ponderada, búsqueda manual de pesos de clase, heurística greedy y técnicas de remuestreo.

El objetivo del repositorio es facilitar la consulta y la reproducibilidad de la metodología utilizada para generar los resultados presentados en la memoria.

## Estructura del repositorio

El repositorio mantiene la estructura de carpetas utilizada durante el desarrollo del trabajo. Esta estructura es importante porque los scripts utilizan rutas relativas y generan los resultados dentro de la carpeta donde se encuentra cada código.

```text
TFM_WBCE_Desbalanceo/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── libs/
│   └── imbalanced_greedy_logreg-0.1.0.tar.gz
│
├── Estrategia_WBCE_Datasets_Sinteticos/
│   └── Codigo_WBCE_Datasets_Sinteticos.py
│
├── Estrategia_WBCE_Dataset_Real_1/
│   ├── Codigo_WBCE_Dataset_Real_1_Cuantitativas_Categoricas.py
│   └── DF1_HBU_JuanSoler.xlsx
│
├── Estrategia_WBCE_Dataset_Real_2/
│   ├── Codigo_WBCE_Dataset_Real_2.py
│   └── SuicideRisk_SecundarySchool_Spain_2025.xlsx
│
└── Estrategias_Clasicas_Desbalanceo/
    ├── Comparaciones_Sinteticos/
    │   └── Comparaciones_Sinteticos.py
    │
    ├── Comparaciones_Real_1_Cuanti_Categ/
    │   └── Comparaciones_Real_1_Cuanti_Categ.py
    │
    └── Comparaciones_Real_2/
        └── Comparaciones_Real_2.py
```

## Importancia de la estructura de carpetas

Para ejecutar correctamente los códigos, se recomienda mantener la estructura de carpetas indicada. Los scripts están diseñados para localizar archivos y guardar resultados mediante rutas relativas a la ubicación del propio código.

En los escenarios reales, los datasets deben colocarse dentro de la misma carpeta que el script correspondiente:

- `DF1_HBU_JuanSoler.xlsx` debe estar en `Estrategia_WBCE_Dataset_Real_1/`.
- `SuicideRisk_SecundarySchool_Spain_2025.xlsx` debe estar en `Estrategia_WBCE_Dataset_Real_2/`.

Si los datasets reales están sujetos a restricciones de uso, confidencialidad o permisos de distribución, no deben subirse a un repositorio público. En ese caso, deben añadirse manualmente en la carpeta correspondiente antes de ejecutar los códigos.

## Escenarios analizados

El repositorio incluye los códigos correspondientes a los escenarios finalmente utilizados en la memoria:

- Datasets sintéticos.
- Dataset Real 1 con variables cuantitativas y categóricas.
- Dataset Real 2 para los objetivos `Suicide_Risk1` y `Suicide_Risk2`.

En el escenario Real 1 se trabaja únicamente con la versión final del dataset que incorpora variables cuantitativas y categóricas codificadas. La versión basada solo en variables cuantitativas fue descartada durante el desarrollo metodológico del trabajo y no se incluye en el repositorio final.

## Estrategias analizadas

Las estrategias incluidas en el trabajo son:

- Regresión logística con pesos igualitarios.
- Rejilla manual de pesos de clase.
- Ponderación automática de Scikit-learn.
- Heurística greedy.
- Random Oversampling.
- Random Undersampling.
- SMOTE.

## Orden de ejecución

Los scripts de comparación dependen de los resultados generados por los códigos base. Por tanto, el orden recomendado de ejecución es el siguiente:

1. Ejecutar `Estrategia_WBCE_Datasets_Sinteticos/Codigo_WBCE_Datasets_Sinteticos.py`.
2. Ejecutar `Estrategia_WBCE_Dataset_Real_1/Codigo_WBCE_Dataset_Real_1_Cuantitativas_Categoricas.py`.
3. Ejecutar `Estrategia_WBCE_Dataset_Real_2/Codigo_WBCE_Dataset_Real_2.py`.
4. Ejecutar `Estrategias_Clasicas_Desbalanceo/Comparaciones_Sinteticos/Comparaciones_Sinteticos.py`.
5. Ejecutar `Estrategias_Clasicas_Desbalanceo/Comparaciones_Real_1_Cuanti_Categ/Comparaciones_Real_1_Cuanti_Categ.py`.
6. Ejecutar `Estrategias_Clasicas_Desbalanceo/Comparaciones_Real_2/Comparaciones_Real_2.py`.

Los códigos base generan las carpetas de salida necesarias, entre ellas:

- `outputs_datasets_sinteticos/`.
- `outputs_dataset_real_1_cuantitativas_categoricas/`.
- `outputs_dataset_real_2_particion_clasica_risk1_risk2/`.

Los scripts de comparación leen esas salidas y generan sus propias carpetas de resultados.

## Instalación de dependencias

Se recomienda crear un entorno virtual de Python antes de instalar las dependencias.

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Después, instalar las dependencias:

```bash
pip install -r requirements.txt
```

La heurística greedy se incluye como una librería local dentro de la carpeta `libs/`, ya que los scripts de comparación dependen de `GreedyClassWeightLogisticRegressionCV`.

## Ejecución de los scripts

Para ejecutar un experimento, acceder a la carpeta correspondiente y lanzar el script principal. Por ejemplo:

```bash
cd Estrategia_WBCE_Dataset_Real_1
python Codigo_WBCE_Dataset_Real_1_Cuantitativas_Categoricas.py
```

Para el escenario Real 2:

```bash
cd Estrategia_WBCE_Dataset_Real_2
python Codigo_WBCE_Dataset_Real_2.py
```

Para las comparaciones, una vez generados los outputs base:

```bash
cd Estrategias_Clasicas_Desbalanceo/Comparaciones_Real_1_Cuanti_Categ
python Comparaciones_Real_1_Cuanti_Categ.py
```

## Salidas generadas

Los scripts generan automáticamente carpetas de salida con resultados, métricas, predicciones, coeficientes, tiempos de ejecución y figuras. Estas carpetas no se incluyen por defecto en el control de versiones, ya que pueden ocupar mucho espacio y se pueden regenerar ejecutando los códigos.

## Nota sobre reproducibilidad

Para reproducir completamente los resultados, es necesario mantener la estructura de carpetas, instalar las dependencias indicadas y ejecutar primero los códigos base WBCE antes de ejecutar los scripts de comparación. Las semillas utilizadas están fijadas en los propios scripts para favorecer la reproducibilidad de los experimentos.

## Autor

Juan Soler Caparrós  
Trabajo Fin de Máster  
Máster Universitario en Inteligencia Artificial
