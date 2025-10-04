# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable 'default payment next month' corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta 'files/input/'.
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna 'default payment next month' a 'default'.
# - Remueva la columna 'ID'.
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría 'others'.
# - Renombre la columna 'default payment next month' a 'default'
# - Remueva la columna 'ID'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como 'files/models/model.pkl.gz'.
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {'predicted_0': 15562, 'predicte_1': 666}, 'true_1': {'predicted_0': 3333, 'predicted_1': 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {'predicted_0': 15562, 'predicte_1': 650}, 'true_1': {'predicted_0': 2490, 'predicted_1': 1420}}
#
# homework.py
# flake8: noqa

# homework.py
# flake8: noqa

# homework.py
# flake8: noqa

# homework.py
# flake8: noqa: E501

import os
import gzip
import pickle
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)

def main():
    x_train = pd.read_pickle('files/grading/x_train.pkl')
    y_train = pd.read_pickle('files/grading/y_train.pkl')
    x_test = pd.read_pickle('files/grading/x_test.pkl')
    y_test = pd.read_pickle('files/grading/y_test.pkl')

    # Pipeline
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    num_cols = [c for c in x_train.columns if c not in cat_cols]

    ct = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols),
    ])

    pipeline = Pipeline([
        ('transform', ct),
        ('pca', PCA()),
        ('selectk', SelectKBest(score_func=f_classif)),
        ('svc', SVC())  
    ])

    param_grid = {
        'pca__n_components': [None, 10, 20, 30],
        'selectk__k': [10, 15, 20, 25, 'all'],
        'svc__kernel': ['rbf', 'linear', 'poly'],
        'svc__C': [0.1, 1, 10, 100, 1000, 5000, 10000],
        'svc__gamma': ['scale', 'auto']
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='balanced_accuracy',
        cv=cv,
        n_jobs=-1
    )

    search.fit(x_train, y_train)
    best_model = search

    os.makedirs('files/models', exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(best_model, f)

    y_pred_train = best_model.predict(x_train)
    y_pred_test = best_model.predict(x_test)

    train_precision = precision_score(y_train, y_pred_train)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    train_recall = recall_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train)

    test_precision = precision_score(y_test, y_pred_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_pred_train).ravel()
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_pred_test).ravel()

    os.makedirs('files/output', exist_ok=True)
    with open('files/output/metrics.json', 'w', encoding='utf-8') as f:
        m_train = {
            'type': 'metrics', 'dataset': 'train',
            'precision': train_precision,
            'balanced_accuracy': train_bal_acc,
            'recall': train_recall,
            'f1_score': train_f1
        }
        f.write(json.dumps(m_train) + '\n')

        m_test = {
            'type': 'metrics', 'dataset': 'test',
            'precision': test_precision,
            'balanced_accuracy': test_bal_acc,
            'recall': test_recall,
            'f1_score': test_f1
        }
        f.write(json.dumps(m_test) + '\n')

        cm_train = {
            'type': 'cm_matrix', 'dataset': 'train',
            'true_0': {'predicted_0': int(tn_tr), 'predicted_1': int(fp_tr)},
            'true_1': {'predicted_0': int(fn_tr), 'predicted_1': int(tp_tr)},
        }
        f.write(json.dumps(cm_train) + '\n')

        cm_test = {
            'type': 'cm_matrix', 'dataset': 'test',
            'true_0': {'predicted_0': int(tn_te), 'predicted_1': int(fp_te)},
            'true_1': {'predicted_0': int(fn_te), 'predicted_1': int(tp_te)},
        }
        f.write(json.dumps(cm_test) + '\n')


if __name__ == '__main__':
    main()