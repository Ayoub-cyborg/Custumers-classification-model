'''
This is the churn_library.py python file that we used  to find custumers who are likely to churn 
The execution of this file will produce artefacts in images and modedls folders.
Date: 02/05/2023
'''

import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import joblib

# Pour la journalisation
logging.basicConfig(
    filename='./logs/churns_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

# Importation des données


def import_data(path):
    df = pd.read_csv(path)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].apply(lambda val: 0 if val == "No" else 1)

    return df


def data_spliting(df):
    train, test = train_test_split(
        df, test_size=0.3, random_state=123, stratify=df['Churn'])
    test, validate = train_test_split(
        test, test_size=0.5, random_state=123, stratify=test['Churn'])

    # Enregistrement des diff ensemble de données
    train.to_csv('./data/train.csv', index=False)
    validate.to_csv('./data/validate.csv', index=False)
    test.to_csv('./data/test.csv', index=False)

    X_train, X_val = train.drop(
        'Churn', axis=1), validate.drop(
        'Churn', axis=1)
    y_train, y_val = train['Churn'], validate['Churn']

    return train, X_train, y_train, X_val, y_val


def perform_eda(df):

    df_copy = df.copy()

    list_columns = df_copy.columns.to_list()

    list_columns.append('Heatmap')

    df_corr = df_copy.corr(numeric_only=True)

    for column_name in list_columns:
        plt.figure(figsize=(10, 6))
        if column_name == 'Heatmap':
            sns.heatmap(
                df_corr,
                mask=np.triu(np.ones_like(df_corr, dtype=bool)),
                center=0, cmap='RdBu', linewidths=1, annot=True,
                fmt=".2f", vmin=-1, vmax=1
            )
        else:
            if df[column_name].dtype != '0':
                df[column_name].hist()
            else:
                sns.countplot(data=df, x=column_name)
        plt.savefig("images/eda/" + column_name + ".jpg")
        plt.close()


def classification_report_image(y_train, y_train_preds, y_val, y_val_preds):
    class_reports_dico = {
        "Logistic Regression train results": classification_report(
            y_train,
            y_train_preds),
        "Logistic Regression validataion results": classification_report(
            y_val,
            y_val_preds
        )
    }

    for title, report in class_reports_dico.items():
        plt.rc('figure', figsize=(7, 3))
        plt.text(
            0.2, 0.3, str(report), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.title(title, fontweight='bold')
        plt.savefig("images/results/" + title + ".jpg")
        plt.close()


# Fonction pour regler le probleme de la colonne 'TotalCharges'
def convert_totalcharges(x):
    # X : dataframe
    Z = x.copy()
    Z['TotalCharges'] = pd.to_numeric(Z['TotalCharges'], errors='coerce')
    return Z.values


def build_pipeline():
    numeric_features = [
        'SeniorCitizen',
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]

    categorical_features = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod']
    # Pipeline de prétraitement des variables independantes numérique
    numeric_transformer = Pipeline(
        steps=[('convert', FunctionTransformer(convert_totalcharges)),
               ('imputer', SimpleImputer(strategy='median')),
               ('scaler', StandardScaler())]
    )

    # Pipeline de prétraitement des variables independantes qualitative
    categorical_transformer = Pipeline(
        steps=[
            ('onehotencoder',
             OneHotEncoder(
                 sparse_output=False,
                 handle_unknown='ignore'))])

    # Combinaison des deux précédents pipelines en un seul
    preprocessor = ColumnTransformer(
        transformers=[('numeric',
                       numeric_transformer,
                       numeric_features),
                      ('categorical',
                       categorical_transformer,
                       categorical_features)]
    )

    # Construction du pipeline de modélisation avec l'algorithme régression
    # liniére comme estimateur
    pipeline_model = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('logreg', LogisticRegression(solver='newton-cg',
                                             random_state=123,
                                             max_iter=2000,
                                             C=5.0
                                             ))]
    )

    return pipeline_model


def train_models(X_train, X_val, y_train, y_val):
    # Formation du modéle
    model = build_pipeline()
    model.fit(X_train, y_train)

    # Prédiction
    y_train_preds_lr = model.predict(X_train)
    y_val_preds_lr = model.predict(X_val)

    # ROC curves images
    RocCurveDisplay.from_estimator(model, X_val, y_val)
    plt.savefig("images/results/roc_curve.jpg")
    plt.close()

    # Classification repports images
    classification_report_image(
        y_train,
        y_train_preds_lr,
        y_val,
        y_val_preds_lr)

    # Enregistrement du modele
    joblib.dump(model, './Models/logreg_model.pkl')


def main():
    logging.info("Data Importation...")
    raw_data = import_data("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    logging.info("Data Importation : SUCCES")

    logging.info("Data Spliting...")
    train_data, Xtrain, ytrain, Xval, yval = data_spliting(raw_data)
    logging.info("Data Spliting : SUCCES")

    logging.info("Exploratory Data Analysis...")
    perform_eda(train_data)
    logging.info("Exploratory Data Analysis : SUCCES")

    logging.info("Training Model...")
    train_models(Xtrain, Xval, ytrain, yval)
    logging.info("Training Model : SUCCES")


if __name__ == "__main__":
    print("Running...")
    main()
    print("completed successfully!!")
