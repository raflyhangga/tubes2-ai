import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.gaussian_nb import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB as SklearnGNB

def load_train_data(base_path: str) -> pd.DataFrame:
    print("Loading training data...")
    df_train_additional = pd.read_csv(f'{base_path}/train/additional_features_train.csv')
    df_train_basic = pd.read_csv(f'{base_path}/train/basic_features_train.csv')
    df_train_content = pd.read_csv(f'{base_path}/train/content_features_train.csv')
    df_train_flow = pd.read_csv(f'{base_path}/train/flow_features_train.csv')
    df_train_labels = pd.read_csv(f'{base_path}/train/labels_train.csv')
    df_train_time = pd.read_csv(f'{base_path}/train/time_features_train.csv')

    df_train = df_train_additional.merge(
        df_train_basic, on='id', how='outer'
    ).merge(
        df_train_content, on='id', how='outer'
    ).merge(
        df_train_flow, on='id', how='outer'
    ).merge(
        df_train_labels, on='id', how='outer'
    ).merge(
        df_train_time, on='id', how='outer'
    )
    return df_train

def load_test_data(base_path: str) -> pd.DataFrame:
    print("Loading test data...")
    df_test_additional = pd.read_csv(f'{base_path}/test/additional_features_test.csv')
    df_test_basic = pd.read_csv(f'{base_path}/test/basic_features_test.csv')
    df_test_content = pd.read_csv(f'{base_path}/test/content_features_test.csv')
    df_test_flow = pd.read_csv(f'{base_path}/test/flow_features_test.csv')
    df_test_time = pd.read_csv(f'{base_path}/test/time_features_test.csv')

    df_test = df_test_additional.merge(
        df_test_basic, on='id', how='outer'
    ).merge(
        df_test_content, on='id', how='outer'
    ).merge(
        df_test_flow, on='id', how='outer'
    ).merge(
        df_test_time, on='id', how='outer'
    )
    return df_test

def prepare_train_test_features(df_train: pd.DataFrame, df_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df_train.drop_duplicates()
    if df_test is not None:
        df_test = df_test.drop_duplicates()
    
    X_train = df_train.drop(['attack_cat', 'id', 'label'], axis=1)
    
    if df_test is not None:
        X_test = df_test.drop(['id'], axis=1)
    else:
        X_test = None
    
    return X_train, X_test

def preprocess_data(X_train: pd.DataFrame, y_train: pd.Series = None, 
                   X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    
    for col in numeric_columns:
        median_value = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_value)
        if X_test is not None:
            X_test[col] = X_test[col].fillna(median_value)
    
    for col in categorical_columns:
        mode_value = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_value)
        if X_test is not None:
            X_test[col] = X_test[col].fillna(mode_value)
    
    for col in categorical_columns:
        train_values = set(X_train[col].unique())
        test_values = set(X_test[col].unique()) if X_test is not None else set()
        all_values = sorted(train_values.union(test_values))
        
        le = LabelEncoder()
        le.fit(all_values)
        
        X_train[col] = le.transform(X_train[col])
        if X_test is not None:
            X_test[col] = le.transform(X_test[col])
    
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    if X_test is not None:
        X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    return X_train, X_test

def compare_models(X_train: np.ndarray, X_test: np.ndarray, 
                  y_train: np.ndarray, y_test: np.ndarray,
                  attack_categories: List[str]) -> Tuple[GaussianNaiveBayes, SklearnGNB]:
    gnb_scratch = GaussianNaiveBayes()
    gnb_sklearn = SklearnGNB()
    
    print("Training scratch implementation...")
    gnb_scratch.fit(X_train, y_train)
    y_pred_scratch = gnb_scratch.predict(X_test)
    score_scratch = gnb_scratch.score(X_test, y_test)
    
    print("Training sklearn implementation...")
    gnb_sklearn.fit(X_train, y_train)
    y_pred_sklearn = gnb_sklearn.predict(X_test)
    score_sklearn = gnb_sklearn.score(X_test, y_test)
    
    print("\nAccuracy Comparison:")
    print(f"Scratch Implementation: {score_scratch:.4f}")
    print(f"Sklearn Implementation: {score_sklearn:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(confusion_matrix(y_test, y_pred_scratch), 
                annot=True, fmt='d', ax=ax1,
                xticklabels=attack_categories,
                yticklabels=attack_categories)
    ax1.set_title('Confusion Matrix - Scratch Implementation')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    sns.heatmap(confusion_matrix(y_test, y_pred_sklearn), 
                annot=True, fmt='d', ax=ax2,
                xticklabels=attack_categories,
                yticklabels=attack_categories)
    ax2.set_title('Confusion Matrix - Sklearn Implementation')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\nClassification Report - Scratch Implementation:")
    print(classification_report(y_test, y_pred_scratch, target_names=attack_categories))
    
    print("\nClassification Report - Sklearn Implementation:")
    print(classification_report(y_test, y_pred_sklearn, target_names=attack_categories))
    
    return gnb_scratch, gnb_sklearn

def save_predictions(model: GaussianNaiveBayes, X_test: np.ndarray, 
                    ids: np.ndarray, filename: str) -> None:
    y_pred = model.predict(X_test)
    df_predictions = pd.DataFrame({
        'id': ids,
        'predicted_attack': y_pred
    })
    df_predictions.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def main():
    base_path = '../data'
    
    df_train = load_train_data(base_path)
    df_test = load_test_data(base_path)
    
    attack_categories = sorted(df_train['attack_cat'].unique())
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['attack_cat'])
    
    X_train_raw, X_test_raw = prepare_train_test_features(df_train, df_test)
    
    print("Preprocessing data...")
    X_train_processed, X_test_processed = preprocess_data(X_train_raw, y_train, X_test_raw)
    
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_processed.values, y_train, 
        test_size=0.2, random_state=42, stratify=y_train
    )
    
    print("Comparing models...")
    gnb_scratch, _ = compare_models(X_train, X_val, y_train_split, y_val, attack_categories)
    
    print("Saving scratch model...")
    gnb_scratch.save_model('gnb_scratch_model.pkl')

if __name__ == "__main__":
    main()