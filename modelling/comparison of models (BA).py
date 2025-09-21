import os
import sys
import numpy as np
import cupy as cp
import kagglehub
import pandas as pd
import xgboost as xgb

# --- RAPIDS IMPORTS ---
import cudf
from cuml.model_selection import train_test_split, GridSearchCV
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
# NOTE: We will use RandomForestClassifier(n_estimators=1) to act as a Decision Tree
from cuml.metrics import accuracy_score, confusion_matrix, roc_auc_score
from cuml.preprocessing import StandardScaler

# --- SCIKIT-LEARN IMPORTS (for CPU-based tasks) ---
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold as SklearnStratifiedKFold

# --- PLOTTING (CPU-based) ---
import seaborn as sns
import matplotlib.pyplot as plt


# --- Configuration ---
SELECTED_COLUMNS = [
    'months', 'marital', 'adults', 'hnd_price', 'new_cell', 'ownrent',
    'dwlltype', 'lor', 'dwllsize', 'infobase', 'numbcars', 'HHstatin',
    'forgntvl', 'ethnic', 'kid0_2', 'kid3_5', 'kid6_10', 'kid11_15',
    'kid16_17', 'creditcd', 'eqpdays', 'rev_Mean', 'mou_Mean', 'totmrc_Mean',
    'change_mou', 'change_rev', 'churn'
]

# --- Data Loading (GPU) ---
def load_and_preprocess_data_gpu():
    """
    Downloads data via Kaggle Hub, loads it into a GPU DataFrame (cudf),
    and performs preprocessing steps like handling missing values and encoding.
    """
    print("Step 1: Locating dataset via Kaggle Hub (will use cache if available)...")
    dataset_folder_path = kagglehub.dataset_download("abhinav89/telecom-customer")
    csv_file_path = os.path.join(dataset_folder_path, "Telecom_customer churn.csv")

    print(f"Loading data from: {csv_file_path}")
    gdf = cudf.read_csv(csv_file_path)
    selected_columns_present = [col for col in SELECTED_COLUMNS if col in gdf.columns]
    gdf_selected = gdf[selected_columns_present].copy()

    print("Step 2: Handling missing values and encoding on GPU...")
    y = gdf_selected['churn'].astype('int32')
    X = gdf_selected.drop('churn', axis=1)

    numerical_cols = X.select_dtypes(include=np.number).columns.to_list()
    categorical_cols = X.select_dtypes(include=['object']).columns.to_list()

    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median()).astype('float32')

    if categorical_cols:
         for col in categorical_cols:
             mode_val = X[col].mode().iloc[0]
             X[col] = X[col].fillna(mode_val)

    X = cudf.get_dummies(X, columns=categorical_cols, drop_first=True, dtype='float32')
    print("Data loaded and preprocessed on GPU.")
    return X, y

# --- Model Training & Evaluation Logic (GPU) ---
def evaluate_model_performance_gpu(model, X_test: cudf.DataFrame, y_test: cudf.Series) -> dict:
    """Calculates performance metrics for a trained model."""
    y_pred = model.predict(X_test)
    if isinstance(y_pred, cudf.Series):
        y_pred = y_pred.to_cupy()

    y_prob_output = model.predict_proba(X_test)
    if isinstance(y_prob_output, cudf.DataFrame):
        y_prob = y_prob_output[1]
    else: # It's a numpy/cupy array from XGBoost
        y_prob = y_prob_output[:, 1]

    if isinstance(y_prob, cudf.Series):
        y_prob = y_prob.to_cupy()

    y_test_cpu = y_test.to_numpy()
    y_prob_cpu = cp.asnumpy(y_prob)
    fpr, tpr, _ = roc_curve(y_test_cpu, y_prob_cpu)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr, "tpr": tpr,
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

def train_and_evaluate_original_models_gpu(X_train, X_test, y_train, y_test):
    """Trains and evaluates baseline models on the GPU."""
    print("\n--- Training and Evaluating Original Models on GPU ---")
    models = {
        "Logistic Regression": LogisticRegression(solver='qn', tol=1e-3),
        # THIS IS THE FIX: A Random Forest with 1 estimator is a Decision Tree
        "Decision Tree": RandomForestClassifier(n_estimators=1, max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42), # Default n_estimators is 100
        "Gradient Boosting (XGB)": xgb.XGBClassifier(device='cuda', eval_metric='logloss', random_state=42)
    }
    results = {}
    for name, model in models.items():
        print(f"Training Original {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model_performance_gpu(model, X_test, y_test)
    return results

def cuml_accuracy_scorer(estimator, X, y_true):
    """Scorer function compatible with cuML GridSearchCV."""
    y_pred = estimator.predict(X)
    return accuracy_score(y_true, y_pred)

def train_and_evaluate_tuned_models_gpu(X_train, X_test, y_train, y_test):
    """Finds best hyperparameters using GPU-accelerated GridSearchCV."""
    print("\n--- Tuning Models on GPU with GridSearchCV ---")
    models = {
        "Logistic Regression": LogisticRegression(),
        # THIS IS THE FIX: A Random Forest with 1 estimator is a Decision Tree
        "Decision Tree": RandomForestClassifier(n_estimators=1, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting (XGB)": xgb.XGBClassifier(device='cuda', eval_metric='logloss', random_state=42)
    }
    param_grids = {
        "Logistic Regression": {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'qn'],
            'tol': [1e-4, 1e-3]
        },
        # Grid for our "Decision Tree" - we only tune tree-specific parameters
        "Decision Tree": {
            'max_depth': [5, 10, 20],
            'min_samples_leaf': [5, 10, 20]
            # 'n_estimators' is NOT tuned, it's fixed at 1
        },
        "Random Forest": {
            'n_estimators': [100, 200], # Here we tune the number of trees
            'max_depth': [10, 20],
            'min_samples_leaf': [2, 4]
        },
        "Gradient Boosting (XGB)": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }
    results = {}
    y_train_cpu = y_train.to_numpy()

    for name, model in models.items():
        print(f"Tuning {name}...")
        sklearn_cv_splitter = SklearnStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            model,
            param_grids[name],
            cv=sklearn_cv_splitter,
            scoring=cuml_accuracy_scorer
        )
        grid_search.fit(X_train, y_train_cpu)
        print(f"Best Parameters for {name}: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        results[name] = evaluate_model_performance_gpu(best_model, X_test, y_test)
    return results

# --- Visualization ---
def plot_visualizations(results, model_type, split_output_dir, split_ratio_str):
    """Generates and saves plots. Moves data from GPU to CPU for plotting."""
    fig_cm, axes = plt.subplots(1, len(results), figsize=(24, 5), squeeze=False)
    axes = axes.flatten()
    fig_cm.suptitle(f'Confusion Matrices for {model_type} Models ({split_ratio_str} Split)', fontsize=16)
    for i, (name, result) in enumerate(results.items()):
        conf_matrix_cpu = result["confusion_matrix"].get()
        sns.heatmap(conf_matrix_cpu, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(split_output_dir, f'confusion_matrices_{model_type.lower()}.png'))
    plt.close(fig_cm)

    fig_roc = plt.figure(figsize=(10, 8))
    for name, result in results.items():
        plt.plot(result["fpr"], result["tpr"], lw=2, label=f'{name} (AUC = {result["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve for {model_type} Models ({split_ratio_str} Split)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(split_output_dir, f'roc_curve_{model_type.lower()}.png'))
    plt.close(fig_roc)

# --- Logging ---
def log_model_results(log_file_path, original_results, tuned_results, split_ratio_str):
    """
    Writes a clean summary of model performance metrics to a specified log file.
    """
    with open(log_file_path, 'w') as f:
        f.write(f"--- Performance Summary for {split_ratio_str} Train/Test Split ---\n\n")

        f.write("--- Original Models ---\n")
        for name, metrics in original_results.items():
            f.write(f"Model: {name}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write("  Confusion Matrix:\n")
            conf_matrix_cpu = metrics['confusion_matrix'].get()
            conf_matrix_df = pd.DataFrame(conf_matrix_cpu, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
            f.write(f"{conf_matrix_df.to_string()}\n\n")

        f.write("\n--- Tuned Models ---\n")
        for name, metrics in tuned_results.items():
            f.write(f"Model: {name}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write("  Confusion Matrix:\n")
            conf_matrix_cpu = metrics['confusion_matrix'].get()
            conf_matrix_df = pd.DataFrame(conf_matrix_cpu, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
            f.write(f"{conf_matrix_df.to_string()}\n\n")


# --- Main Execution (Sequential) ---
def main():
    """Main function to run the sequential GPU analysis for different train/test splits."""
    output_dir = 'output_gpu'
    os.makedirs(output_dir, exist_ok=True)
    X, y = load_and_preprocess_data_gpu()
    train_sizes = [0.6, 0.7, 0.8, 0.9]
    all_results_summary = []

    print(f"\n{'='*80}\nStarting SEQUENTIAL analysis for splits: {train_sizes}\n{'='*80}")

    for train_size in train_sizes:
        train_ratio = int(train_size * 100)
        test_ratio = 100 - train_ratio
        split_ratio_str = f"{train_ratio}_{test_ratio}"
        split_output_dir = os.path.join(output_dir, split_ratio_str)
        os.makedirs(split_output_dir, exist_ok=True)

        print(f"\n--- Starting analysis for {split_ratio_str} Split ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42, stratify=y
        )

        print("Applying StandardScaler (Z-score) to numerical features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        original_results = train_and_evaluate_original_models_gpu(X_train_scaled, X_test_scaled, y_train, y_test)
        tuned_results = train_and_evaluate_tuned_models_gpu(X_train_scaled, X_test_scaled, y_train, y_test)

        log_file_path = os.path.join(split_output_dir, 'analysis_output.txt')
        log_model_results(log_file_path, original_results, tuned_results, split_ratio_str)
        print(f"Performance summary for {split_ratio_str} saved to '{log_file_path}'")

        plot_visualizations(original_results, "Original", split_output_dir, split_ratio_str)
        plot_visualizations(tuned_results, "Tuned", split_output_dir, split_ratio_str)
        print(f"Visualizations for {split_ratio_str} saved in '{split_output_dir}/'")

        split_summary = []
        for name in original_results.keys():
            split_summary.append({
                "Split Ratio": split_ratio_str, "Model": name, "Type": "Original",
                "Accuracy": float(original_results[name]['accuracy']),
                "AUC": float(original_results[name]['roc_auc'])
            })
            split_summary.append({
                "Split Ratio": split_ratio_str, "Model": name, "Type": "Tuned",
                "Accuracy": float(tuned_results[name]['accuracy']),
                "AUC": float(tuned_results[name]['roc_auc'])
            })
        all_results_summary.extend(split_summary)

    print(f"\n{'='*80}\n--- FINAL COMPREHENSIVE SUMMARY ACROSS ALL SPLITS (GPU) ---\n{'='*80}")
    final_summary_df = pd.DataFrame(all_results_summary)
    pivot_df_acc = final_summary_df.pivot_table(
        index=['Model', 'Type'], columns='Split Ratio', values='Accuracy'
    ).sort_index(level=0, sort_remaining=False)
    print("Accuracy Scores:")
    print(pivot_df_acc.to_string(float_format="%.4f"))

    pivot_df_auc = final_summary_df.pivot_table(
        index=['Model', 'Type'], columns='Split Ratio', values='AUC'
    ).sort_index(level=0, sort_remaining=False)
    print("\nAUC Scores:")
    print(pivot_df_auc.to_string(float_format="%.4f"))

    final_summary_path = os.path.join(output_dir, 'final_comprehensive_summary.txt')
    with open(final_summary_path, 'w') as f:
        f.write("--- Final Comprehensive Summary Across All Splits (GPU Workflow) ---\n\n")
        f.write("Accuracy Scores:\n")
        f.write(pivot_df_acc.to_string(float_format="%.4f"))
        f.write("\n\nAUC Scores:\n")
        f.write(pivot_df_auc.to_string(float_format="%.4f"))

    print(f"\nFinal summary saved to {final_summary_path}")
    print("GPU script finished successfully.")

if __name__ == "__main__":
    main()