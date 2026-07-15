# Descripción

Este proyecto es una **aplicación de Machine Learning para clasificación**. El dataset utilizado es  [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), con el objetivo de predecir el comportamiento para retener a la clientela.

El modelo desarrollado utiliza el dataset de recurso, puede de predecir la satisfacción de los clientes y posteriormente se productiviza esta solución en una aplicación.

El objetivo es que las personas de un departamento de marketing tengan una herramienta que facillite la tome de decisiones comerciales. 

### Propósito de la aplicación
La aplicación resuelve el problema de **automatizar la clasificación de casos** a partir de variables de entrada (features). En lugar de decidir manualmente, el sistema genera una predicción de clase de forma consistente y reproducible.

### Problema que soluciona
- Reduce el tiempo y esfuerzo de análisis manual.
- Ofrece una decisión basada en datos y patrones aprendidos por el modelo.
- Facilita que el usuario obtenga un resultado de forma guiada (UI + pipeline de predicción).

### Público objetivo
- Equipo analítico / negocio que necesita clasificar casos.
- Usuarios que requieren una predicción rápida con una interfaz simple.
- Sistemas que quieran integrar predicciones con una entrada estructurada.

## Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-007bff?style=for-the-badge&labelColor=007bff&color=007bff&logo=python&logoColor=ffffff)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-28a745?style=for-the-badge&labelColor=28a745&color=28a745&logo=ai&logoColor=ffffff)
![Docker](https://img.shields.io/badge/Docker-ffc107?style=for-the-badge&labelColor=ffc107&color=ffc107&logo=docker&logoColor=000000)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-ff4507?style=for-the-badge&labelColor=ff4507&color=ff4507&logo=postgresql&logoColor=000000)
![SQLAlchemy](https://img.shields.io/badge/sqlalchemy-3407ff?style=for-the-badge&labelColor=3407ff&color=3407ff&logo=sqlalchemy&logoColor=000000)
![Jupyter](https://img.shields.io/badge/Jupyter-ff07e6?style=for-the-badge&labelColor=ff07e6&color=ff07e6&logo=jupyter&logoColor=000000)
![NumPy](https://img.shields.io/badge/NumPy-079cff?style=for-the-badge&labelColor=079cff&color=079cff&logo=numpy&logoColor=000000)
![Pandas](https://img.shields.io/badge/Pandas-ff8b07?style=for-the-badge&labelColor=ff8b07&color=ff8b07&logo=pandas&logoColor=000000)

         

# Estructura del código

```
proyecto5-grupo5/
|__ data
   |__ processed
   |   |__ telco_clean.parquet
   |__ raw
|__ models
   |__ baseline.pkl
   |__ logistic_regression.pkl
   |__ random_forest.pkl
   |__ knn_pipeline.joblib
   |__ logistic_pipeline.joblib
   |__ random_forest_pipeline.joblib
   |__ xgboost_pipeline.joblib
|__ notebooks
   |__ 01_eda.ipynb
   |__ 01_eda.py
   |__ 02_logistic_regression.ipynb
   |__ 02_logistic_regression.py
   |__ 03_random_forest.ipynb
   |__ 03_random_forest.py
   |__ 04_knn.ipynb
   |__ 04_knn.py
   |__ 05_xgboost.ipynb
   |__ 05_xgboost.py
|__ reports
   |__ eda_report.html
|__ src
   |__ __init__.py
   |__ config.py
   |__ data.py
   |__ eda.py
   |__ evaluation.py
   |__ features.py
   |__ pipelines.py
   |__ preprocessing.py
   |__ training.py
   |__ utils.py
   |__ proyecto5_grupo5.egg-info
|__ frontend
   |__ src
      |__ api
         |__ client.ts
         |__ mock.ts
         |__ predict.ts
      |__ components
         |__ FormField.tsx
         |__ FormSection.tsx
         |__ LanguageToggle.tsx
         |__ PredictionResult.tsx
         |__ ThemeToggle.tsx
      |__ context
         |__ ThemeContext.tsx
      |__ data
         |__ fields.ts
      |__ hooks
         |__ usePredictForm.ts
      |__ i18n
         |__ en.ts
         |__ es.ts
         |__ index.tsx
         |__ types.ts
      |__ lib
         |__ logger.ts
         |__ utils.ts
      |__ pages
         |__ PredictPage.tsx
      |__ types
         |__ api.ts
         |__ ui.ts
      |__ App.tsx
      |__ index.css
      |__ main.tsx
      |__ vite-env.d.ts
|__ proyecto5_grupo5.egg-info



```
## Uso de servidores de despliegue 
La app está pensada para poder desplegarse como:
- **Frontend** (React) 
- **Backend y predicción**
- 
### Diseño de la base de datos
Actualmente, el dataset se gestiona principalmente mediante archivos y el pipeline trabaja con un dataset **preprocesado**.

### Documentación general del proyecto y APIs
- La lógica del pipeline se documenta mediante módulos en `src/`.
- La interfaz web se documenta mediante componentes en `frontend/src/`.
- Hay un flujo de predicción implementado en la capa de frontend (`predict`) y/o integración con el backend.
# Cómo ejecutar el proyecto (guía rápida)

### 1) Entrenamiento / Generación de artefactos
- Los notebooks `notebooks/0x_*.ipynb` muestran el proceso de EDA y entrenamiento.
- Los modelos finales se guardan en `models/` y el dataset limpio en `data/processed/`.

### 2) Uso / Predicción
- Ejecutar el frontend para consumir la predicción.
- Los modelos/pipelines se reutilizan desde `models/` (según la implementación).

## Próximos pasos 
- Completar despliegue y documentar el link.
- Añadir captura/ejemplo real del flujo de predicción.
- Incorporar métricas finales y gráfica de evaluación en `reports/`.
- Documentar API (si aplica) o el mecanismo exacto de llamada desde frontend.
- Incorporar una base de datos