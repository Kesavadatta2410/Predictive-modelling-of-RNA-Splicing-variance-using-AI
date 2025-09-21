import two
#!/usr/bin/env python3
"""
Gene Expression Classifier Workflow: Steps 13-22
Class Imbalance Handling, Calibration, Explainability, Visualization, 
Interpretation, Reproducibility, and Extensions to Splicing Analysis

Author: AI Research Pipeline
Date: September 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Advanced ML and Evaluation Libraries
from sklearn.metrics import (
    brier_score_loss, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import permutation_test
import scipy.stats as stats

# Class Imbalance Handling
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")
    IMBLEARN_AVAILABLE = False

# Model Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

# Pathway Analysis
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    print("Warning: GSEApy not available. Install with: pip install gseapy")
    GSEAPY_AVAILABLE = False

# Statistical Tests
from scipy.stats import bootstrap
from sklearn.model_selection import permutation_test_score

# ============================================================================
# STEP 13: HANDLING CLASS IMBALANCE
# ============================================================================

def assess_class_imbalance(y_train, y_val=None, y_test=None):
    """
    Comprehensive assessment of class imbalance across all splits
    """
    
    print(f"\n=== CLASS IMBALANCE ASSESSMENT ===")
    
    splits = {'Train': y_train}
    if y_val is not None:
        splits['Validation'] = y_val
    if y_test is not None:
        splits['Test'] = y_test
    
    imbalance_info = {}
    
    for split_name, y_data in splits.items():
        class_counts = y_data.value_counts()
        total_samples = len(y_data)
        
        # Calculate imbalance ratio
        majority_count = class_counts.max()
        minority_count = class_counts.min()
        imbalance_ratio = majority_count / minority_count
        
        print(f"\n{split_name} Set:")
        print(f"  Total samples: {total_samples}")
        print(f"  Class distribution:")
        for class_label, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"    {class_label}: {count} ({percentage:.1f}%)")
        
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Imbalance severity assessment
        if imbalance_ratio < 1.5:
            severity = "Balanced"
        elif imbalance_ratio < 3:
            severity = "Mild imbalance"
        elif imbalance_ratio < 10:
            severity = "Moderate imbalance"
        else:
            severity = "Severe imbalance"
        
        print(f"  Severity: {severity}")
        
        imbalance_info[split_name] = {
            'class_counts': class_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'severity': severity
        }
    
    return imbalance_info

def apply_imbalance_techniques(X_train, y_train, method='smote', random_state=42):
    
    
    print(f"\n=== APPLYING IMBALANCE TECHNIQUE: {method.upper()} ===")
    
    if not IMBLEARN_AVAILABLE:
        print("Error: imbalanced-learn not available. Returning original data.")
        return X_train, y_train, {}
    
    original_counts = y_train.value_counts()
    print(f"Original class distribution: {original_counts.to_dict()}")
    
    # Choose technique
    techniques = {
        'smote': SMOTE(random_state=random_state),
        'borderline_smote': BorderlineSMOTE(random_state=random_state),
        'svm_smote': SVMSMOTE(random_state=random_state),
        'adasyn': ADASYN(random_state=random_state),
        'smote_tomek': SMOTETomek(random_state=random_state),
        'smote_enn': SMOTEENN(random_state=random_state)
    }
    
    if method not in techniques:
        print(f"Unknown method '{method}'. Using 'smote'.")
        method = 'smote'
    
    try:
        # Apply technique
        technique = techniques[method]
        X_resampled, y_resampled = technique.fit_resample(X_train, y_train)
        
        # Convert back to pandas if needed
        if isinstance(X_train, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        if isinstance(y_train, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        resampled_counts = pd.Series(y_resampled).value_counts()
        print(f"Resampled class distribution: {resampled_counts.to_dict()}")
        
        # Calculate sampling statistics
        sampling_stats = {
            'original_samples': len(X_train),
            'resampled_samples': len(X_resampled),
            'samples_added': len(X_resampled) - len(X_train),
            'original_distribution': original_counts.to_dict(),
            'resampled_distribution': resampled_counts.to_dict(),
            'method': method
        }
        
        print(f"Samples added: {sampling_stats['samples_added']}")
        print(f"Total samples: {len(X_train)} → {len(X_resampled)}")
        
        # Save resampling info
        with open('results/imbalance_strategy.txt', 'w') as f:
            f.write(f"Imbalance Handling Strategy: {method}\n")
            f.write(f"Original samples: {len(X_train)}\n")
            f.write(f"Resampled samples: {len(X_resampled)}\n")
            f.write(f"Method: {method}\n")
            for key, value in sampling_stats.items():
                f.write(f"{key}: {value}\n")
        
        # Create visualization
        create_imbalance_visualization(original_counts, resampled_counts, method)
        
        print(f"✓ {method.upper()} applied successfully")
        
        return X_resampled, y_resampled, sampling_stats
        
    except Exception as e:
        print(f"Error applying {method}: {e}")
        return X_train, y_train, {}

def create_imbalance_visualization(original_counts, resampled_counts, method):
    """Create visualization showing before/after imbalance handling"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original distribution
    ax1.bar(range(len(original_counts)), original_counts.values, 
           color=['skyblue', 'lightcoral'])
    ax1.set_xticks(range(len(original_counts)))
    ax1.set_xticklabels(original_counts.index)
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Original Class Distribution')
    ax1.grid(alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(original_counts.values):
        ax1.text(i, v + max(original_counts.values) * 0.01, str(v), 
                ha='center', va='bottom')
    
    # Resampled distribution
    ax2.bar(range(len(resampled_counts)), resampled_counts.values,
           color=['lightgreen', 'orange'])
    ax2.set_xticks(range(len(resampled_counts)))
    ax2.set_xticklabels(resampled_counts.index)
    ax2.set_ylabel('Number of Samples')
    ax2.set_title(f'After {method.upper()}')
    ax2.grid(alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(resampled_counts.values):
        ax2.text(i, v + max(resampled_counts.values) * 0.01, str(v), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'figs/imbalance_handling_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STEP 14: CALIBRATION & RISK SCORE
# ============================================================================

def calibrate_model_probabilities(model, X_train, y_train, X_val, y_val, 
                                 method='platt', cv=5):
    """
    Calibrate model probabilities using Platt scaling or isotonic regression
    """
    
    print(f"\n=== PROBABILITY CALIBRATION ===")
    print(f"Method: {method}")
    
    # Get uncalibrated predictions
    if hasattr(model, 'predict_proba'):
        y_val_proba_uncal = model.predict_proba(X_val)[:, 1]
    else:
        print("Model doesn't support probability prediction. Skipping calibration.")
        return model, {}
    
    # Calculate uncalibrated metrics
    uncal_brier = brier_score_loss(y_val, y_val_proba_uncal)
    
    print(f"Uncalibrated Brier score: {uncal_brier:.4f}")
    
    # Calibrate the model
    calibration_methods = {'platt': 'sigmoid', 'isotonic': 'isotonic'}
    calib_method = calibration_methods.get(method, 'sigmoid')
    
    try:
        calibrated_model = CalibratedClassifierCV(
            model, method=calib_method, cv=cv
        )
        calibrated_model.fit(X_train, y_train)
        
        # Get calibrated predictions
        y_val_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
        
        # Calculate calibrated metrics
        cal_brier = brier_score_loss(y_val, y_val_proba_cal)
        
        print(f"Calibrated Brier score: {cal_brier:.4f}")
        print(f"Brier score improvement: {uncal_brier - cal_brier:.4f}")
        
        # Calibration statistics
        calibration_stats = {
            'method': method,
            'uncalibrated_brier': uncal_brier,
            'calibrated_brier': cal_brier,
            'brier_improvement': uncal_brier - cal_brier,
            'cv_folds': cv
        }
        
        # Create calibration plots
        create_calibration_plots(y_val, y_val_proba_uncal, y_val_proba_cal, method)
        
        # Save calibration results
        with open('results/calibration_results.txt', 'w') as f:
            for key, value in calibration_stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Model calibration complete")
        
        return calibrated_model, calibration_stats
        
    except Exception as e:
        print(f"Error during calibration: {e}")
        return model, {}

def create_calibration_plots(y_true, y_prob_uncal, y_prob_cal, method):
    """Create comprehensive calibration analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calibration curve - uncalibrated
    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_prob_uncal, n_bins=10)
    
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[0, 0].plot(mean_pred_uncal, fraction_pos_uncal, 's-', 
                   label=f'Uncalibrated (Brier: {brier_score_loss(y_true, y_prob_uncal):.3f})')
    axes[0, 0].set_xlabel('Mean Predicted Probability')
    axes[0, 0].set_ylabel('Fraction of Positives')
    axes[0, 0].set_title('Calibration Curve - Uncalibrated')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Calibration curve - calibrated
    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_prob_cal, n_bins=10)
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[0, 1].plot(mean_pred_cal, fraction_pos_cal, 's-', color='green',
                   label=f'Calibrated (Brier: {brier_score_loss(y_true, y_prob_cal):.3f})')
    axes[0, 1].set_xlabel('Mean Predicted Probability')
    axes[0, 1].set_ylabel('Fraction of Positives')
    axes[0, 1].set_title(f'Calibration Curve - {method.title()}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Probability distribution histograms
    axes[1, 0].hist(y_prob_uncal, bins=20, alpha=0.7, density=True, 
                   label='Uncalibrated')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Probability Distribution - Uncalibrated')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(y_prob_cal, bins=20, alpha=0.7, density=True, 
                   color='green', label='Calibrated')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title(f'Probability Distribution - {method.title()}')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figs/calibration_analysis_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STEP 15: EVALUATION & STATISTICAL VALIDATION
# ============================================================================

def comprehensive_model_evaluation(model, X_test, y_test, n_bootstrap=1000):
    """
    Comprehensive evaluation with confidence intervals and statistical tests
    """
    
    print(f"\n=== COMPREHENSIVE MODEL EVALUATION ===")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None
    
    # Basic metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'sensitivity': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'specificity': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        metrics['pr_auc'] = average_precision_score(y_test, y_proba)
        metrics['brier_score'] = brier_score_loss(y_test, y_proba)
    
    print("Test Set Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Bootstrap confidence intervals
    print(f"\nCalculating bootstrap confidence intervals ({n_bootstrap} samples)...")
    ci_results = calculate_bootstrap_confidence_intervals(
        y_test, y_pred, y_proba, n_bootstrap=n_bootstrap)
    
    # Permutation test
    print("Performing permutation test...")
    perm_results = perform_permutation_test(model, X_test, y_test)
    
    # Save comprehensive results
    eval_results = {
        'test_metrics': metrics,
        'confidence_intervals': ci_results,
        'permutation_test': perm_results
    }
    
    # Save to files
    save_evaluation_results(eval_results)
    
    # Create evaluation plots
    create_evaluation_plots(y_test, y_pred, y_proba, eval_results)
    
    print(f"✓ Comprehensive evaluation complete")
    
    return eval_results

def calculate_bootstrap_confidence_intervals(y_true, y_pred, y_proba=None, 
                                           n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for key metrics"""
    
    from sklearn.utils import resample
    
    def bootstrap_metric(metric_func, *args):
        bootstrap_scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(n_samples), n_samples=n_samples)
            
            try:
                if len(args) == 1:  # Binary predictions only
                    score = metric_func(y_true[indices], args[0][indices])
                else:  # Predictions + probabilities
                    score = metric_func(y_true[indices], args[0][indices], args[1][indices])
                bootstrap_scores.append(score)
            except:
                continue
        
        if len(bootstrap_scores) == 0:
            return None, None
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
        
        return lower, upper
    
    # Convert to numpy for indexing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_proba is not None:
        y_proba = np.array(y_proba)
    
    ci_results = {}
    
    # Metrics using predictions only
    from sklearn.metrics import accuracy_score, f1_score
    
    metrics_pred = {
        'accuracy': accuracy_score,
        'f1': f1_score
    }
    
    for name, func in metrics_pred.items():
        lower, upper = bootstrap_metric(func, y_pred)
        if lower is not None:
            ci_results[name] = {'lower': lower, 'upper': upper}
            print(f"  {name.upper()} 95% CI: [{lower:.3f}, {upper:.3f}]")
    
    # Metrics using probabilities
    if y_proba is not None:
        def auc_wrapper(y_true, y_pred, y_proba):
            return roc_auc_score(y_true, y_proba)
        
        def brier_wrapper(y_true, y_pred, y_proba):
            return brier_score_loss(y_true, y_proba)
        
        prob_metrics = {
            'roc_auc': auc_wrapper,
            'brier_score': brier_wrapper
        }
        
        for name, func in prob_metrics.items():
            lower, upper = bootstrap_metric(func, y_pred, y_proba)
            if lower is not None:
                ci_results[name] = {'lower': lower, 'upper': upper}
                print(f"  {name.upper()} 95% CI: [{lower:.3f}, {upper:.3f}]")
    
    return ci_results

def perform_permutation_test(model, X_test, y_test, n_permutations=1000):
    """Perform permutation test to assess statistical significance"""
    
    try:
        # Perform permutation test
        score, perm_scores, pvalue = permutation_test_score(
            model, X_test, y_test, scoring='roc_auc',
            cv=5, n_permutations=n_permutations, random_state=42
        )
        
        perm_results = {
            'true_score': score,
            'permutation_scores_mean': np.mean(perm_scores),
            'permutation_scores_std': np.std(perm_scores),
            'p_value': pvalue,
            'n_permutations': n_permutations
        }
        
        print(f"  True AUC score: {score:.4f}")
        print(f"  Permutation scores mean: {np.mean(perm_scores):.4f} ± {np.std(perm_scores):.4f}")
        print(f"  P-value: {pvalue:.6f}")
        
        return perm_results
        
    except Exception as e:
        print(f"  Error in permutation test: {e}")
        return {}

def save_evaluation_results(eval_results):
    """Save comprehensive evaluation results"""
    
    # Test metrics CSV
    metrics_df = pd.DataFrame([eval_results['test_metrics']])
    metrics_df.to_csv('results/test_metrics.csv', index=False)
    
    # Confidence intervals
    if eval_results['confidence_intervals']:
        ci_data = []
        for metric, ci in eval_results['confidence_intervals'].items():
            ci_data.append({
                'metric': metric,
                'lower_ci': ci['lower'],
                'upper_ci': ci['upper']
            })
        ci_df = pd.DataFrame(ci_data)
        ci_df.to_csv('results/confidence_intervals.csv', index=False)
    
    # Permutation test results
    if eval_results['permutation_test']:
        with open('results/permutation_test_results.txt', 'w') as f:
            for key, value in eval_results['permutation_test'].items():
                f.write(f"{key}: {value}\n")

def create_evaluation_plots(y_true, y_pred, y_proba, eval_results):
    """Create comprehensive evaluation visualization"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Confusion Matrix
    plt.subplot(3, 4, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        plt.subplot(3, 4, 2)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
    
    # Precision-Recall Curve
    if y_proba is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        plt.subplot(3, 4, 3)
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
    
    # Classification Report Heatmap
    plt.subplot(3, 4, 4)
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T
    sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='RdYlBu', fmt='.3f')
    plt.title('Classification Report')
    
    # Metrics with Confidence Intervals
    plt.subplot(3, 4, 5)
    if 'confidence_intervals' in eval_results and eval_results['confidence_intervals']:
        ci_data = eval_results['confidence_intervals']
        metrics = list(ci_data.keys())
        test_scores = [eval_results['test_metrics'][m] for m in metrics]
        lower_bounds = [ci_data[m]['lower'] for m in metrics]
        upper_bounds = [ci_data[m]['upper'] for m in metrics]
        
        x_pos = range(len(metrics))
        plt.bar(x_pos, test_scores, yerr=[
            [s - l for s, l in zip(test_scores, lower_bounds)],
            [u - s for s, u in zip(test_scores, upper_bounds)]
        ], capsize=5)
        plt.xticks(x_pos, metrics, rotation=45)
        plt.ylabel('Score')
        plt.title('Metrics with 95% Confidence Intervals')
        plt.grid(alpha=0.3)
    
    # Permutation Test Results
    if 'permutation_test' in eval_results and eval_results['permutation_test']:
        perm_data = eval_results['permutation_test']
        
        plt.subplot(3, 4, 6)
        # Create histogram of null distribution
        null_scores = np.random.normal(
            perm_data['permutation_scores_mean'],
            perm_data['permutation_scores_std'],
            1000
        )
        plt.hist(null_scores, bins=50, alpha=0.7, density=True, 
                label='Null Distribution')
        plt.axvline(perm_data['true_score'], color='red', linestyle='--', 
                   label=f"True Score (p={perm_data['p_value']:.4f})")
        plt.xlabel('AUC Score')
        plt.ylabel('Density')
        plt.title('Permutation Test')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STEP 16: EXPLAINABILITY & MAPPING TO BIOLOGY
# ============================================================================

def explain_model_predictions(model, X_train, X_test, feature_names=None, 
                            model_type='tree', top_k=20):
    """
    Generate model explanations using SHAP
    """
    
    print(f"\n=== MODEL EXPLAINABILITY WITH SHAP ===")
    
    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping explainability analysis.")
        return {}
    
    print(f"Model type: {model_type}")
    print(f"Generating explanations for top {top_k} features...")
    
    try:
        # Choose appropriate SHAP explainer
        if model_type == 'tree':
            # For tree-based models (Random Forest, XGBoost, etc.)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary
                
        elif model_type == 'linear':
            # For linear models
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
            
        elif model_type == 'kernel':
            # For any model (slower but universal)
            explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(100))
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
        else:
            print(f"Unknown model type: {model_type}")
            return {}
        
        # Feature names
        if feature_names is None:
            if hasattr(X_test, 'columns'):
                feature_names = X_test.columns.tolist()
            else:
                feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(0)
        top_features_idx = np.argsort(feature_importance)[-top_k:][::-1]
        
        # Get top features and their importance
        top_features = [feature_names[i] for i in top_features_idx]
        top_importance = feature_importance[top_features_idx]
        
        print(f"Top {len(top_features)} most important features:")
        for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        # Create SHAP plots
        create_shap_plots(shap_values, X_test, feature_names, top_features_idx)
        
        # Save results
        top_features_df = pd.DataFrame({
            'feature': top_features,
            'importance': top_importance,
            'rank': range(1, len(top_features) + 1)
        })
        top_features_df.to_csv('results/top_features.csv', index=False)
        
        explanation_results = {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'top_features': top_features,
            'top_importance': top_importance,
            'explainer_type': type(explainer).__name__
        }
        
        print(f"✓ SHAP explanation complete")
        
        return explanation_results
        
    except Exception as e:
        print(f"Error in SHAP explanation: {e}")
        return {}

def create_shap_plots(shap_values, X_test, feature_names, top_features_idx, max_display=20):
    """Create comprehensive SHAP visualization plots"""
    
    try:
        # Limit to top features for visualization
        shap_values_top = shap_values[:, top_features_idx[:max_display]]
        X_test_top = X_test.iloc[:, top_features_idx[:max_display]]
        feature_names_top = [feature_names[i] for i in top_features_idx[:max_display]]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Summary plot
        plt.sca(axes[0, 0])
        shap.summary_plot(shap_values_top, X_test_top, 
                         feature_names=feature_names_top, 
                         plot_type="dot", show=False)
        axes[0, 0].set_title('SHAP Summary Plot (Dot)')
        
        # Bar plot
        plt.sca(axes[0, 1])
        shap.summary_plot(shap_values_top, X_test_top,
                         feature_names=feature_names_top,
                         plot_type="bar", show=False)
        axes[0, 1].set_title('SHAP Feature Importance (Bar)')
        
        # Waterfall plot for first sample
        plt.sca(axes[1, 0])
        if hasattr(shap, 'waterfall_plot'):
            shap.waterfall_plot(
                shap.Explanation(values=shap_values_top[0], 
                               base_values=0,
                               data=X_test_top.iloc[0].values,
                               feature_names=feature_names_top),
                show=False
            )
        axes[1, 0].set_title('SHAP Waterfall Plot (Sample 1)')
        
        # Force plot data preparation
        plt.sca(axes[1, 1])
        mean_shap = np.abs(shap_values_top).mean(0)
        sorted_idx = np.argsort(mean_shap)[-10:]
        
        axes[1, 1].barh(range(len(sorted_idx)), mean_shap[sorted_idx])
        axes[1, 1].set_yticks(range(len(sorted_idx)))
        axes[1, 1].set_yticklabels([feature_names_top[i] for i in sorted_idx])
        axes[1, 1].set_xlabel('Mean |SHAP value|')
        axes[1, 1].set_title('Top 10 Features by Mean |SHAP|')
        
        plt.tight_layout()
        plt.savefig('figs/shap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ SHAP plots saved")
        
    except Exception as e:
        print(f"  Error creating SHAP plots: {e}")

def perform_pathway_enrichment_analysis(top_features, organism='human'):
    """
    Perform pathway enrichment analysis on top features
    """
    
    print(f"\n=== PATHWAY ENRICHMENT ANALYSIS ===")
    
    if not GSEAPY_AVAILABLE:
        print("GSEApy not available. Skipping pathway analysis.")
        return {}
    
    print(f"Analyzing {len(top_features)} top features...")
    
    try:
        # Gene set libraries for enrichment
        gene_set_libraries = [
            'GO_Biological_Process_2023',
            'GO_Molecular_Function_2023', 
            'KEGG_2021_Human',
            'Reactome_2022'
        ]
        
        enrichment_results = {}
        
        for library in gene_set_libraries:
            print(f"  Analyzing {library}...")
            
            try:
                # Perform enrichment analysis
                enr = gp.enrichr(
                    gene_list=top_features,
                    gene_sets=library,
                    organism='Human',
                    outdir=None,
                    cutoff=0.05,
                    no_plot=True
                )
                
                if not enr.results.empty:
                    # Filter significant results
                    significant_results = enr.results[enr.results['Adjusted P-value'] < 0.05]
                    
                    if not significant_results.empty:
                        enrichment_results[library] = significant_results
                        print(f"    Found {len(significant_results)} significant pathways")
                    else:
                        print(f"    No significant pathways found")
                else:
                    print(f"    No results returned")
                    
            except Exception as e:
                print(f"    Error with {library}: {e}")
                continue
        
        # Save enrichment results
        if enrichment_results:
            save_pathway_results(enrichment_results)
            create_pathway_plots(enrichment_results)
            
            print(f"✓ Pathway enrichment analysis complete")
            print(f"  Found enriched pathways in {len(enrichment_results)} databases")
        else:
            print("  No significant pathways found in any database")
        
        return enrichment_results
        
    except Exception as e:
        print(f"Error in pathway analysis: {e}")
        return {}

def save_pathway_results(enrichment_results):
    """Save pathway enrichment results to CSV files"""
    
    for library, results in enrichment_results.items():
        # Clean library name for filename
        clean_name = library.replace('_', '').replace(' ', '').lower()
        filename = f'results/pathway_enrichment_{clean_name}.csv'
        
        # Select key columns
        key_columns = ['Term', 'P-value', 'Adjusted P-value', 'Combined Score', 'Genes']
        available_columns = [col for col in key_columns if col in results.columns]
        
        results[available_columns].to_csv(filename, index=False)

def create_pathway_plots(enrichment_results):
    """Create pathway enrichment visualization"""
    
    if not enrichment_results:
        return
    
    n_libraries = len(enrichment_results)
    fig, axes = plt.subplots(n_libraries, 1, figsize=(12, 4 * n_libraries))
    
    if n_libraries == 1:
        axes = [axes]
    
    for i, (library, results) in enumerate(enrichment_results.items()):
        # Take top 10 most significant pathways
        top_pathways = results.head(10).copy()
        
        # Create horizontal bar plot
        y_pos = range(len(top_pathways))
        scores = -np.log10(top_pathways['Adjusted P-value'])
        
        axes[i].barh(y_pos, scores, alpha=0.7)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels([term[:50] + '...' if len(term) > 50 else term 
                                for term in top_pathways['Term']], fontsize=8)
        axes[i].set_xlabel('-log10(Adjusted P-value)')
        axes[i].set_title(f'{library} - Top Enriched Pathways')
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/pathway_enrichment.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STEP 17: VISUALIZATION & FIGURES FOR REPORT
# ============================================================================

def create_publication_figures(eval_results, explanation_results, expression_data=None):
    """
    Create publication-quality figures for the report
    """
    
    print(f"\n=== CREATING PUBLICATION FIGURES ===")
    
    # Figure 1: Data overview and preprocessing
    if expression_data is not None:
        create_data_overview_figure(expression_data)
    
    # Figure 2: Model performance comparison
    create_model_performance_figure(eval_results)
    
    # Figure 3: Feature importance and biological interpretation
    if explanation_results:
        create_feature_interpretation_figure(explanation_results)
    
    # Figure 4: Model validation and statistical analysis
    create_validation_figure(eval_results)
    
    print("✓ Publication figures created")

def create_data_overview_figure(expression_data):
    """Create comprehensive data overview figure"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Sample distribution
    plt.subplot(2, 3, 1)
    # Add your data visualization code here
    plt.title('Sample Distribution')
    
    # Gene expression distribution
    plt.subplot(2, 3, 2)
    # Add expression distribution plot
    plt.title('Expression Distribution')
    
    # Continue with other subplots...
    plt.tight_layout()
    plt.savefig('figs/figure1_data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_performance_figure(eval_results):
    """Create model performance comparison figure"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROC curves comparison
    # PR curves comparison  
    # Calibration curves
    # Performance metrics with CI
    
    plt.tight_layout()
    plt.savefig('figs/figure2_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_interpretation_figure(explanation_results):
    """Create feature interpretation and biological context figure"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # SHAP summary
    # Top features
    # Pathway enrichment
    # Gene-gene interactions
    
    plt.tight_layout()
    plt.savefig('figs/figure3_feature_interpretation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_validation_figure(eval_results):
    """Create model validation and statistical analysis figure"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cross-validation results
    # Permutation test
    # Bootstrap confidence intervals
    # Calibration analysis
    
    plt.tight_layout()
    plt.savefig('figs/figure4_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STEP 18-22: REMAINING WORKFLOW COMPONENTS
# ============================================================================

def generate_methods_section():
    """Generate Methods section text for publication"""
    
    methods_text = """
## Methods

### Data Collection and Preprocessing
Gene expression data were obtained from [specify source]. Quality control included...

### Feature Engineering
Gene-level features were extracted using... Most variable genes were selected based on...

### Model Development
Baseline models included logistic regression with L1/L2 regularization, random forest, and support vector machines. Deep learning models utilized fully connected neural networks with...

### Class Imbalance Handling
Class imbalance was addressed using [specify method]. The technique generates synthetic samples...

### Model Calibration
Probability predictions were calibrated using [Platt scaling/isotonic regression] to ensure reliable confidence estimates...

### Statistical Validation
Model performance was validated using stratified cross-validation. Statistical significance was assessed using permutation tests with 1000 permutations...

### Explainability Analysis
Model interpretability was achieved using SHAP (Shapley Additive Explanations). Feature importance was calculated...

### Pathway Enrichment Analysis
Top predictive features were analyzed for biological pathway enrichment using GSEApy with GO, KEGG, and Reactome databases...
"""
    
    with open('report/Methods.md', 'w') as f:
        f.write(methods_text)
    
    print("✓ Methods section generated")

def create_reproducibility_package():
    """Create complete reproducibility package"""
    
    print(f"\n=== CREATING REPRODUCIBILITY PACKAGE ===")
    
    # Environment specification
    create_environment_file()
    
    # Analysis notebook
    create_analysis_notebook()
    
    # Model deployment script
    create_deployment_script()
    
    # Documentation
    create_documentation()
    
    print("✓ Reproducibility package created")

def create_environment_file():
    """Create environment specification file"""
    
    requirements = """
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Machine learning
imbalanced-learn>=0.10.0
torch>=1.13.0
xgboost>=1.6.0

# Explainability
shap>=0.41.0

# Pathway analysis
gseapy>=1.0.0

# Statistical analysis
scipy>=1.9.0
statsmodels>=0.13.0

# Utilities
joblib>=1.1.0
tqdm>=4.64.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)

def create_analysis_notebook():
    """Create Jupyter notebook with complete analysis"""
    
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gene Expression Classification Analysis\\n",
    "\\n",
    "Complete reproducible workflow for gene expression-based disease classification."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import required libraries\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "# Add complete import list..."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python", 
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open('notebooks/complete_analysis.ipynb', 'w') as f:
        f.write(notebook_content)

def create_deployment_script():
    """Create model deployment script"""
    
    deployment_code = '''
#!/usr/bin/env python3
"""
Model Deployment Script
Deploy trained models for production use
"""

import joblib
import pandas as pd
import numpy as np

class GeneExpressionClassifier:
    def __init__(self, model_path, scaler_path=None):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
    
    def predict(self, X):
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        return self.model.predict_proba(X)

if __name__ == "__main__":
    # Example usage
    classifier = GeneExpressionClassifier(
        'models/best_model.pkl',
        'models/scaler.pkl'
    )
    
    # Load new data
    new_data = pd.read_csv('new_samples.csv', index_col=0)
    
    # Make predictions
    predictions = classifier.predict(new_data)
    probabilities = classifier.predict_proba(new_data)
    
    print("Predictions:", predictions)
    print("Probabilities:", probabilities)
'''
    
    with open('deploy_model.py', 'w') as f:
        f.write(deployment_code)

def create_documentation():
    """Create comprehensive documentation"""
    
    readme_content = """
# Gene Expression Classification Pipeline

## Overview
This repository contains a complete pipeline for gene expression-based classification with explainable AI.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run analysis: `python main_analysis.py`
3. View results in `results/` and `figs/` directories

## Pipeline Steps
1. Data organization and provenance
2. Gene annotation and filtering  
3. Normalization and batch correction
4. Feature engineering
5. Model training and validation
6. Explainability analysis
7. Pathway enrichment

## Key Features
- Handles class imbalance with SMOTE
- Model calibration for reliable probabilities
- SHAP-based explainability
- Comprehensive statistical validation
- Publication-ready figures

## Repository Structure
- `data/`: Raw and processed data
- `results/`: Analysis results and metrics
- `models/`: Trained models
- `figs/`: Publication figures
- `notebooks/`: Jupyter notebooks

## Citation
If you use this pipeline, please cite: [Your paper]
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)

# ============================================================================
# MAIN EXECUTION FUNCTION FOR STEPS 13-22
# ============================================================================

def execute_advanced_analysis(trained_models, splits, features_dict, **kwargs):
    """
    Execute advanced analysis steps 13-22
    """
    
    print("=" * 80)
    print("ADVANCED ANALYSIS PIPELINE: STEPS 13-22")  
    print("Imbalance → Calibration → Explainability → Publication")
    print("=" * 80)
    
    results = {}
    
    # Get the best model from previous steps
    best_model_name = kwargs.get('best_model', 'Random_Forest')
    best_model = trained_models.get(best_model_name)
    
    if best_model is None:
        print(f"Model {best_model_name} not found. Using first available model.")
        best_model = list(trained_models.values())[0]
    
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val'] 
    X_test, y_test = splits['X_test'], splits['y_test']
    
    # Step 13: Class imbalance handling
    if kwargs.get('handle_imbalance', False):
        imbalance_info = assess_class_imbalance(y_train, y_val, y_test)
        
        if imbalance_info['Train']['imbalance_ratio'] > 2.0:
            X_train_balanced, y_train_balanced, sampling_stats = apply_imbalance_techniques(
                X_train, y_train, method=kwargs.get('imbalance_method', 'smote'))
            results['imbalance_handling'] = sampling_stats
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Step 14: Model calibration
    if kwargs.get('calibrate_model', True):
        calibrated_model, calibration_stats = calibrate_model_probabilities(
            best_model, X_train_balanced, y_train_balanced, X_val, y_val,
            method=kwargs.get('calibration_method', 'platt'))
        results['calibration'] = calibration_stats
        
        # Use calibrated model for remaining analysis
        final_model = calibrated_model if calibration_stats else best_model
    else:
        final_model = best_model
    
    # Step 15: Comprehensive evaluation
    eval_results = comprehensive_model_evaluation(
        final_model, X_test, y_test, 
        n_bootstrap=kwargs.get('n_bootstrap', 1000))
    results['evaluation'] = eval_results
    
    # Step 16: Explainability analysis
    if kwargs.get('explain_model', True):
        explanation_results = explain_model_predictions(
            final_model, X_train, X_test, 
            feature_names=X_train.columns.tolist() if hasattr(X_train, 'columns') else None,
            model_type=kwargs.get('model_type_for_shap', 'tree'))
        results['explanations'] = explanation_results
        
        # Pathway enrichment
        if explanation_results and 'top_features' in explanation_results:
            pathway_results = perform_pathway_enrichment_analysis(
                explanation_results['top_features'])
            results['pathways'] = pathway_results
    
    # Steps 17-22: Publication and reproducibility
    if kwargs.get('create_publication_materials', True):
        # Create publication figures
        create_publication_figures(eval_results, 
                                 results.get('explanations', {}))
        
        # Generate methods section
        generate_methods_section()
        
        # Create reproducibility package
        create_reproducibility_package()
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("ADVANCED ANALYSIS COMPLETE")
    print("=" * 80)
    
    if 'evaluation' in results:
        test_auc = results['evaluation']['test_metrics'].get('roc_auc', 'N/A')
        print(f"✓ Final test AUC: {test_auc}")
    
    if 'explanations' in results:
        n_features = len(results['explanations'].get('top_features', []))
        print(f"✓ Top predictive features identified: {n_features}")
    
    if 'pathways' in results:
        n_pathways = sum(len(v) for v in results['pathways'].values())
        print(f"✓ Enriched pathways found: {n_pathways}")
    
    print(f"\n📁 Complete analysis outputs:")
    print("   - results/test_metrics.csv")
    print("   - results/top_features.csv") 
    print("   - results/pathway_enrichment_*.csv")
    print("   - figs/figure*_*.png")
    print("   - report/Methods.md")
    print("   - notebooks/complete_analysis.ipynb")
    
    return results

# Example usage
if __name__ == "__main__":
    print("Advanced analysis steps 13-22 loaded.")
    print("Use execute_advanced_analysis() with your trained models and data splits.")
    
    # Example parameters
    advanced_params = {
        'best_model': 'Random_Forest',
        'handle_imbalance': True,
        'imbalance_method': 'smote',
        'calibrate_model': True,
        'calibration_method': 'platt', 
        'explain_model': True,
        'model_type_for_shap': 'tree',
        'n_bootstrap': 1000,
        'create_publication_materials': True
    }
    
    # Example call (uncomment to run):
    # advanced_results = execute_advanced_analysis(
    #     trained_models=your_trained_models,
    #     splits=your_data_splits,
    #     features_dict=your_features,
    #     **advanced_params
    # )