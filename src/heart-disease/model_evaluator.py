import pandas as pd
import seaborn as sns
from pathlib import Path
from logger import logger
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

Path("artifacts").mkdir(parents=True, exist_ok=True)

def model_evaluator(X_test: pd.DataFrame, y_test: pd.Series, 
                    le: LabelEncoder, lr_model: LogisticRegression) -> None:
    """
    Evaluates a trained Logistic Regression model.

    Args:
        X_test: pd.DataFrame
        y_test: pd.Series
        le: fitted Labelencoder
        lr_model: trained Logistic Regression Model

    Returns:
        None
    """
    y_pred = lr_model.predict(X_test)
    y_pred_labelled = le.inverse_transform(y_pred)
    y_true_labelled = le.inverse_transform(y_test)
    model_performance = classification_report(y_pred=y_pred_labelled,
                                            y_true=y_true_labelled)
    logger.info("Model evaluation complete")
    print(model_performance)

    model_confusion_matrix = confusion_matrix(y_true=y_true_labelled,
                                          y_pred=y_pred_labelled)
    class_names = le.classes_

    plt.figure(figsize=(8,6))
    sns.heatmap(model_confusion_matrix, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True)
    plt.savefig("artifacts/confusion_matrix.png", bbox_inches='tight')
    logger.info("Confusion matrix plot saved to artifacts")
    
    plt.close()