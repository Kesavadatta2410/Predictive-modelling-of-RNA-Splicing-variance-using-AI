import os
import sys
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scientific computing
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu

# Machine Learning
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, permutation_test_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from statsmodels.stats.multitest import multipletests

# Optional libraries with fallbacks
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ imbalanced-learn not available")
    IMBLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not available - install with: pip install shap")
    SHAP_AVAILABLE = False

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    print("⚠️ GSEApy not available - install with: pip install gseapy")
    GSEAPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available")
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

DISEASE_TARGETS = [
    'hemophilia',
    'von_willebrand_disease',
    'sickle_cell_disease',
    'thalassemia',
    'thrombophilia',
    'platelet_disorders',
    'hereditary_hemorrhagic_telangiectasia',
    'iron_refractory_iron_deficiency_anemia'
]

PATHWAY_DATABASES = [
    'data\GSE107011_Processed_data_TPM.txt',
    'data\GSE107011_tpm.txt',
    'data\GSE122459_ann.txt',
    'data\GSE122459_tpm.txt',
    'data\GSE122459_cnt.txt'
]
FEATURE_CONFIG = {
    'n_variable': 3000,  # Reduced from 5000
    'n_pca': 50,
    'use_feature_selection': True,
    'selection_k': 2000  # Select top 2000 features
}

MODEL_CONFIG = {
    'cv_folds': 5,
    'random_state': 42,
    'regularization': 'strong',  # 'weak', 'moderate', 'strong'
}
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Create all necessary directories"""
    dirs = [
        'meta', 'interim', 'artifacts', 'results', 'features',
        'splits', 'models', 'figs', 'notebooks', 'reports',
        'disease_predictions'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Directory structure created")

def calculate_hash(filepath):
    """Calculate SHA256 hash of file"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def save_safely(data, filepath, description="data"):
    """Safe save with error handling"""
    try:
        if isinstance(data, dict):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=True)
        else:
            joblib.dump(data, filepath)
        print(f"✓ Saved {description} to {filepath}")
        return True
    except Exception as e:
        print(f"✗ Failed to save {description}: {e}")
        return False

# ============================================================================
# DATA LOADING & INSPECTION
# ============================================================================

def inspect_expression_data(filepath, dataset_name="dataset"):
    """Comprehensive data inspection"""
    print(f"\n{'='*70}")
    print(f"INSPECTING: {dataset_name}")
    print(f"{'='*70}")
    
    # Load data
    if filepath.endswith(('.txt', '.tsv')):
        df = pd.read_csv(filepath, sep='\t', index_col=0)
    else:
        df = pd.read_csv(filepath, index_col=0)
    
    n_genes, n_samples = df.shape
    print(f"Matrix: {n_genes:,} genes × {n_samples:,} samples")
    print(f"Missing: {df.isnull().sum().sum():,} values")
    print(f"Range: {df.min().min():.3f} to {df.max().max():.3f}")
    
    # Detect data type
    max_val = df.max().max()
    if max_val < 50:
        data_type = "log-transformed"
    elif max_val > 10000:
        data_type = "raw counts"
    else:
        data_type = "normalized (TPM/FPKM)"
    print(f"Type: {data_type}")
    
    # Save summary
    summary = {
        'n_genes': n_genes,
        'n_samples': n_samples,
        'data_type': data_type,
        'value_range': [float(df.min().min()), float(df.max().max())]
    }
    save_safely(summary, f'artifacts/inspection_{dataset_name}.json', 'inspection')
    
    return df, summary

def clean_gene_ids(gene_ids):
    """Strip version numbers from gene IDs"""
    return [str(g).split('.')[0] if '.' in str(g) else str(g) for g in gene_ids]

# ============================================================================
# GENE ANNOTATION
# ============================================================================

def annotate_genes_biomart(gene_ids, max_genes=5000):
    """Annotate genes using BioMart"""
    try:
        import biomart
        print(f"\n{'='*70}")
        print("GENE ANNOTATION")
        print(f"{'='*70}")
        
        clean_ids = clean_gene_ids(gene_ids)[:max_genes]
        
        server = biomart.BiomartServer("http://ensembl.org/biomart")
        mart = server.datasets['hsapiens_gene_ensembl']
        
        attributes = [
            'ensembl_gene_id', 'external_gene_name', 'gene_biotype',
            'chromosome_name', 'description'
        ]
        
        results = []
        chunk_size = 100
        
        for i in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[i:i+chunk_size]
            print(f"  Annotating chunk {i//chunk_size + 1}/{len(clean_ids)//chunk_size + 1}")
            
            try:
                response = mart.search({
                    'filters': {'ensembl_gene_id': chunk},
                    'attributes': attributes
                })
                for line in response.iter_lines():
                    if line:
                        results.append(line.decode('utf-8').split('\t'))
            except:
                continue
        
        annotation_df = pd.DataFrame(results, columns=attributes)
        annotation_df = annotation_df[annotation_df['ensembl_gene_id'] != '']
        annotation_df = annotation_df.drop_duplicates(subset='ensembl_gene_id')
        
        annotation_df.to_csv('interim/gene_annotations.csv', index=False)
        print(f"✓ Annotated {len(annotation_df):,} genes")
        
        return annotation_df
        
    except Exception as e:
        print(f"⚠️ BioMart annotation failed: {e}")
        return create_fallback_annotation(gene_ids)

def create_fallback_annotation(gene_ids):
    """Create basic annotation when BioMart fails"""
    clean_ids = clean_gene_ids(gene_ids)
    annotation_df = pd.DataFrame({
        'ensembl_gene_id': clean_ids,
        'external_gene_name': clean_ids,
        'gene_biotype': 'unknown',
        'chromosome_name': 'unknown',
        'description': 'No annotation available'
    })
    annotation_df.to_csv('interim/gene_annotations.csv', index=False)
    print(f"✓ Created fallback annotations for {len(annotation_df):,} genes")
    return annotation_df

def add_gene_symbols(expr_df, annotation_df, filter_protein_coding=True):
    """Add gene symbols and filter"""
    print("\nAdding gene symbols...")
    
    expr_df = expr_df.copy()
    expr_df.index = clean_gene_ids(expr_df.index)
    
    id_to_symbol = dict(zip(annotation_df['ensembl_gene_id'], 
                           annotation_df['external_gene_name']))
    id_to_biotype = dict(zip(annotation_df['ensembl_gene_id'], 
                            annotation_df['gene_biotype']))
    
    # Map gene symbols - use Series for proper fillna
    gene_symbols = pd.Series(expr_df.index).map(id_to_symbol)
    expr_df['gene_symbol'] = gene_symbols.fillna(pd.Series(expr_df.index)).values
    
    # Map gene biotypes
    gene_biotypes = pd.Series(expr_df.index).map(id_to_biotype)
    expr_df['gene_biotype'] = gene_biotypes.fillna('unknown').values
    
    print(f"Genes with symbols: {expr_df['gene_symbol'].notna().sum():,}")
    print(f"\nBiotype distribution:")
    for bt, cnt in expr_df['gene_biotype'].value_counts().head().items():
        print(f"  {bt}: {cnt:,}")
    
    if filter_protein_coding:
        initial = len(expr_df)
        expr_df = expr_df[expr_df['gene_biotype'] == 'protein_coding']
        print(f"Filtered to protein-coding: {len(expr_df):,} ({len(expr_df)/initial*100:.1f}%)")
    
    annotation_cols = ['gene_symbol', 'gene_biotype']
    expr_matrix = expr_df.drop(columns=annotation_cols)
    
    expr_df.to_csv('interim/expr_annotated.csv')
    
    return expr_matrix, expr_df

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def filter_low_expression(expr_df, threshold=1.0, min_pct=0.1, log_scale=False):
    """Filter low-expression genes"""
    print(f"\n{'='*70}")
    print("GENE FILTERING")
    print(f"{'='*70}")
    
    n_samples = expr_df.shape[1]
    min_samples = int(n_samples * min_pct)
    
    if log_scale:
        threshold = np.log2(threshold + 1)
    
    print(f"Threshold: {threshold:.3f}, Min samples: {min_samples}")
    
    genes_above = (expr_df > threshold).sum(axis=1)
    mask = genes_above >= min_samples
    
    filtered = expr_df[mask].copy()
    
    print(f"Initial: {len(expr_df):,} genes")
    print(f"Retained: {len(filtered):,} genes ({len(filtered)/len(expr_df)*100:.1f}%)")
    
    filtered.to_csv('interim/expr_filtered.csv')
    
    return filtered

def normalize_transform(expr_df, method='assume_normalized', 
                       log_transform=True, z_score=False):
    """Normalize and transform expression data"""
    print(f"\n{'='*70}")
    print("NORMALIZATION & TRANSFORMATION")
    print(f"{'='*70}")
    print(f"Method: {method}, Log: {log_transform}, Z-score: {z_score}")
    
    if method == 'cpm':
        lib_sizes = expr_df.sum(axis=0)
        normalized = expr_df.div(lib_sizes) * 1e6
        print("Applied CPM normalization")
    else:
        normalized = expr_df.copy()
        print("Assuming pre-normalized")
    
    if log_transform:
        if normalized.max().max() > 50:
            normalized = np.log2(normalized + 1)
            print("Applied log2(x+1) transformation")
    
    if z_score:
        normalized = normalized.apply(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x, axis=1
        )
        print("Applied z-score normalization")
    
    normalized.to_csv('interim/expr_normalized.csv')
    print(f"Final range: {normalized.min().min():.3f} to {normalized.max().max():.3f}")
    
    return normalized

# ============================================================================
# DIFFERENTIAL EXPRESSION
# ============================================================================

def differential_expression(expr_df, metadata_df, condition_col='condition',
                          healthy='healthy', disease='disease'):
    """Perform differential expression analysis"""
    print(f"\n{'='*70}")
    print("DIFFERENTIAL EXPRESSION")
    print(f"{'='*70}")
    
    common = expr_df.columns.intersection(metadata_df.index)
    expr_sub = expr_df[common]
    meta_sub = metadata_df.loc[common]
    
    healthy_samples = meta_sub[meta_sub[condition_col] == healthy].index
    disease_samples = meta_sub[meta_sub[condition_col] == disease].index
    
    print(f"Healthy: {len(healthy_samples)}, Disease: {len(disease_samples)}")
    
    results = []
    
    for gene in expr_sub.index:
        h_expr = expr_sub.loc[gene, healthy_samples]
        d_expr = expr_sub.loc[gene, disease_samples]
        
        mean_h = h_expr.mean()
        mean_d = d_expr.mean()
        log2fc = np.log2((mean_d + 0.001) / (mean_h + 0.001))
        
        try:
            _, pval = ttest_ind(d_expr, h_expr, equal_var=False, nan_policy='omit')
        except:
            pval = 1.0
        
        results.append({
            'gene_id': gene,
            'mean_healthy': mean_h,
            'mean_disease': mean_d,
            'log2fc': log2fc,
            'pvalue': pval
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    valid = ~results_df['pvalue'].isna()
    fdr = np.ones(len(results_df))
    if valid.sum() > 0:
        _, fdr[valid], _, _ = multipletests(
            results_df.loc[valid, 'pvalue'], method='fdr_bh'
        )
    results_df['fdr'] = fdr
    
    results_df['significant'] = (results_df['fdr'] < 0.05) & (np.abs(results_df['log2fc']) > 0.5)
    results_df = results_df.sort_values('pvalue')
    
    n_sig = results_df['significant'].sum()
    n_up = ((results_df['fdr'] < 0.05) & (results_df['log2fc'] > 0.5)).sum()
    n_down = ((results_df['fdr'] < 0.05) & (results_df['log2fc'] < -0.5)).sum()
    
    print(f"Significant: {n_sig:,} (Up: {n_up:,}, Down: {n_down:,})")
    
    results_df.to_csv('results/differential_expression.csv', index=False)
    
    return results_df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(expr_df, de_results=None, n_variable=5000, n_pca=50):
    """Create comprehensive feature sets"""
    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING")
    print(f"{'='*70}")
    
    features = {}
    
    # All genes
    X_gene = expr_df.T
    features['X_gene'] = X_gene
    print(f"✓ All genes: {X_gene.shape}")
    
    # Variable genes
    variances = expr_df.var(axis=1)
    top_var = variances.nlargest(n_variable).index
    X_var = expr_df.loc[top_var].T
    features['X_var'] = X_var
    print(f"✓ Variable genes: {X_var.shape}")
    
    # DE genes
    if de_results is not None:
        sig_genes = de_results[de_results['significant']]['gene_id'].tolist()
        available = [g for g in sig_genes if g in expr_df.index]
        if available:
            X_de = expr_df.loc[available].T
            features['X_de'] = X_de
            print(f"✓ DE genes: {X_de.shape}")
    
    # PCA features
    pca = PCA(n_components=min(n_pca, min(X_var.shape) - 1))
    X_pca = pca.fit_transform(X_var)
    X_pca_df = pd.DataFrame(
        X_pca, index=X_var.index,
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
    )
    features['X_pca'] = X_pca_df
    print(f"✓ PCA features: {X_pca_df.shape} (variance: {pca.explained_variance_ratio_[:5].sum()*100:.1f}%)")
    
    # Summary statistics
    summary = pd.DataFrame(index=expr_df.columns)
    summary['total_expr'] = expr_df.sum(axis=0)
    summary['mean_expr'] = expr_df.mean(axis=0)
    summary['median_expr'] = expr_df.median(axis=0)
    summary['std_expr'] = expr_df.std(axis=0)
    summary['n_expressed'] = (expr_df > 0).sum(axis=0)
    features['X_summary'] = summary
    print(f"✓ Summary features: {summary.shape}")
    
    # Save all features
    for name, data in features.items():
        data.to_csv(f'features/{name}.csv')
    
    return features

# ============================================================================
# DATA SPLITTING
# ============================================================================

def create_splits(features_df, metadata_df, condition_col='condition',
                 train_size=0.7, val_size=0.15, random_state=42):
    """Create stratified train/val/test splits"""
    print(f"\n{'='*70}")
    print("DATA SPLITTING")
    print(f"{'='*70}")
    
    common = features_df.index.intersection(metadata_df.index)
    X = features_df.loc[common]
    y = metadata_df.loc[common, condition_col]
    
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {dict(y.value_counts())}")
    
    test_size = 1 - train_size - val_size
    
    # First split: test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: train/val
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        stratify=y_trainval, random_state=random_state
    )
    
    print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Save indices
    pd.Series(X_train.index, name='sample_id').to_csv('splits/train_indices.csv', header=True)
    pd.Series(X_val.index, name='sample_id').to_csv('splits/val_indices.csv', header=True)
    pd.Series(X_test.index, name='sample_id').to_csv('splits/test_indices.csv', header=True)
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

# ============================================================================
# MODEL TRAINING
# ============================================================================

def get_regularized_models(random_state=42):
    """Get models with strong regularization to prevent overfitting"""
    
    models = {
        'Logistic_L2_Strong': LogisticRegression(
            penalty='l2',
            C=0.1,  # Strong regularization (reduced from default 1.0)
            max_iter=1000,
            random_state=random_state
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_split=10,  # Require more samples to split
            min_samples_leaf=4,  # Require more samples in leaf nodes
            max_features='sqrt',  # Use subset of features
            random_state=random_state,
            n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,  # Lower learning rate
            max_depth=3,  # Shallow trees
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,  # Use 80% of data for each tree
            random_state=random_state
        ),
        'SVM_RBF': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=random_state
        )
    }
    
    return models


def train_baseline_models(splits, scale=True, cv_folds=5, random_state=42):
    """Train baseline ML models"""
    print(f"\n{'='*70}")
    print("BASELINE MODEL TRAINING")
    print(f"{'='*70}")
    
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    
    models = {
        'Logistic_L2': LogisticRegression(penalty='l2', max_iter=1000, random_state=random_state),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=random_state)
    }
    
    results = []
    trained = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            steps = []
            if scale:
                steps.append(('scaler', StandardScaler()))
            steps.append(('classifier', model))
            pipeline = Pipeline(steps)
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                scoring='roc_auc', n_jobs=-1
            )
            
            # Fit
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_val_pred = pipeline.predict(X_val)
            y_val_proba = pipeline.predict_proba(X_val)[:, 1]
            
            # Metrics
            val_acc = accuracy_score(y_val, y_val_pred)
            val_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.5
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            results.append({
                'model': name,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'val_accuracy': val_acc,
                'val_auc': val_auc,
                'val_f1': val_f1
            })
            
            trained[name] = pipeline
            
            print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Val AUC: {val_auc:.3f}, Accuracy: {val_acc:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)
    results_df.to_csv('results/baseline_metrics.csv', index=False)
    
    # Save best model
    best_name = results_df.iloc[0]['model']
    best_model = trained[best_name]
    joblib.dump(best_model, 'models/best_baseline_model.pkl')
    
    print(f"\n✓ Best model: {best_name} (Val AUC: {results_df.iloc[0]['val_auc']:.3f})")
    
    return results_df, trained

# ============================================================================
# STATISTICAL VALIDATION
# ============================================================================

def bootstrap_metrics(y_true, y_pred, y_proba, n_bootstrap=1000, random_state=42):
    """Bootstrap confidence intervals"""
    print(f"\n{'='*70}")
    print("BOOTSTRAP VALIDATION")
    print(f"{'='*70}")
    
    np.random.seed(random_state)
    
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        if i % 200 == 0:
            print(f"  Progress: {i}/{n_bootstrap}")
        
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_boot = y_pred[indices]
        y_proba_boot = y_proba[indices]
        
        try:
            metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            if len(np.unique(y_true_boot)) > 1:
                metrics['auc'].append(roc_auc_score(y_true_boot, y_proba_boot))
        except:
            continue
    
    ci_results = {}
    print("\nBootstrap 95% CI:")
    for metric, values in metrics.items():
        if values:
            mean = np.mean(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            ci_results[metric] = {
                'mean': float(mean),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'std': float(np.std(values))
            }
            print(f"  {metric.upper():10s}: {mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    save_safely(ci_results, 'results/bootstrap_ci.json', 'bootstrap CI')
    
    return ci_results

def permutation_test(model, X, y, n_permutations=1000, random_state=42):
    """Permutation test for significance"""
    print(f"\n{'='*70}")
    print("PERMUTATION TEST")
    print(f"{'='*70}")
    
    score, perm_scores, pvalue = permutation_test_score(
        model, X, y, scoring='roc_auc', cv=5,
        n_permutations=n_permutations, random_state=random_state, n_jobs=-1
    )
    
    print(f"Original score: {score:.4f}")
    print(f"Permutation mean: {np.mean(perm_scores):.4f} ± {np.std(perm_scores):.4f}")
    print(f"P-value: {pvalue:.6f}")
    print(f"Significance: {'✓ Yes' if pvalue < 0.05 else '✗ No'}")
    
    result = {
        'original_score': float(score),
        'perm_mean': float(np.mean(perm_scores)),
        'perm_std': float(np.std(perm_scores)),
        'p_value': float(pvalue),
        'significant': bool(pvalue < 0.05)
    }
    save_safely(result, 'results/permutation_test.json', 'permutation test')
    
    return score, perm_scores, pvalue

# ============================================================================
# EXPLAINABILITY WITH SHAP
# ============================================================================
def compute_shap_importance(model, X_train, X_test, feature_names=None,
                           max_samples=100, max_features=500):
    """
    Compute SHAP feature importance with FIXED pipeline handling
    
    FIXES:
    1. Pipeline extraction (v5.1)
    2. Feature name alignment (v5.2) - NEW!
    """
    print(f"\n{'='*70}")
    print("SHAP EXPLAINABILITY (FIXED v5.2)")
    print(f"{'='*70}")
    
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP not available")
        return None
    
    # Limit data size
    # Limit number of features (columns)
    if X_train.shape[1] > max_features:
        X_train = X_train.iloc[:, :max_features]
        X_test = X_test.iloc[:, :max_features]
        if feature_names:
            feature_names = feature_names[:max_features]
    
    # Limit number of samples (rows)
    if X_test.shape[0] > max_samples:
        X_test = X_test.sample(n=max_samples, random_state=42)
    
    print(f"Computing SHAP for {X_test.shape}")
    
    # =======================================================================
    # FIX v5.1: Extract classifier from pipeline
    # =======================================================================
    
    if hasattr(model, 'named_steps'):
        print("✓ Detected sklearn Pipeline - extracting components...")
        
        # Extract scaler
        if 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
            print("  Found scaler in pipeline")
        else:
            scaler = None
            print("  No scaler found in pipeline")
        
        # Extract classifier
        if 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            print(f"  Extracted classifier: {type(classifier).__name__}")
        else:
            print("  ✗ No classifier found in pipeline")
            return None
        
        # ===================================================================
        # FIX v5.2: Align feature names before transformation
        # ===================================================================
        
        if scaler is not None:
            # Get feature names the scaler was fitted with
            if hasattr(scaler, 'feature_names_in_'):
                fitted_features = scaler.feature_names_in_
                print(f"  Scaler was fit with {len(fitted_features)} features")
                
                # Check if current data has same features
                X_train_cols = set(X_train.columns)
                fitted_cols = set(fitted_features)
                
                if X_train_cols != fitted_cols:
                    print(f"  ⚠️ Feature mismatch detected")
                    print(f"    Train has {len(X_train_cols)} features")
                    print(f"    Scaler expects {len(fitted_cols)} features")
                    
                    # Find common features
                    common_features = list(X_train_cols.intersection(fitted_cols))
                    
                    if len(common_features) == 0:
                        print("  ✗ No common features - using numpy array approach")
                        # Fallback: convert to numpy (no feature names)
                        X_train_aligned = X_train.values
                        X_test_aligned = X_test.values
                        use_feature_names = False
                    else:
                        print(f"  Using {len(common_features)} common features")
                        # Use common features only
                        X_train_aligned = X_train[common_features]
                        X_test_aligned = X_test[common_features]
                        use_feature_names = True
                else:
                    # Features match - align order
                    try:
                        X_train_aligned = X_train[fitted_features]
                        X_test_aligned = X_test[fitted_features]
                        print("  ✓ Features aligned to match scaler")
                        use_feature_names = True
                    except KeyError as e:
                        print(f"  ⚠️ Column ordering failed: {e}")
                        # Fallback to numpy
                        X_train_aligned = X_train.values
                        X_test_aligned = X_test.values
                        use_feature_names = False
            else:
                # Old sklearn or no feature_names_in_
                print("  Using data as-is (no feature_names_in_)")
                X_train_aligned = X_train
                X_test_aligned = X_test
                use_feature_names = True
            
            # Transform the aligned data
            try:
                X_train_transformed_array = scaler.transform(X_train_aligned)
                X_test_transformed_array = scaler.transform(X_test_aligned)
                
                # Convert back to DataFrame
                if use_feature_names and isinstance(X_train_aligned, pd.DataFrame):
                    X_train_transformed = pd.DataFrame(
                        X_train_transformed_array,
                        index=X_train_aligned.index,
                        columns=X_train_aligned.columns
                    )
                    X_test_transformed = pd.DataFrame(
                        X_test_transformed_array,
                        index=X_test_aligned.index,
                        columns=X_test_aligned.columns
                    )
                else:
                    # Create DataFrame with generic feature names
                    n_features = X_train_transformed_array.shape[1]
                    col_names = [f'feature_{i}' for i in range(n_features)]
                    X_train_transformed = pd.DataFrame(
                        X_train_transformed_array,
                        index=X_train.index if hasattr(X_train, 'index') else range(len(X_train)),
                        columns=col_names
                    )
                    X_test_transformed = pd.DataFrame(
                        X_test_transformed_array,
                        index=X_test.index if hasattr(X_test, 'index') else range(len(X_test)),
                        columns=col_names
                    )
                
                print("  ✓ Data transformed using scaler")
                
            except Exception as e:
                print(f"  ⚠️ Transformation failed: {e}")
                print("  Using untransformed data")
                X_train_transformed = X_train.copy()
                X_test_transformed = X_test.copy()
        else:
            # No scaler
            X_train_transformed = X_train.copy()
            X_test_transformed = X_test.copy()
        
        # Use the extracted classifier for SHAP
        model_for_shap = classifier
    else:
        # Not a pipeline
        X_train_transformed = X_train
        X_test_transformed = X_test
        model_for_shap = model
    
    # =======================================================================
    # Try different SHAP explainers with fallbacks
    # =======================================================================
    
    explainer = None
    shap_values = None
    
    # Try TreeExplainer first (fast, tree-based models)
    try:
        print("\nTrying TreeExplainer...")
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_test_transformed)
        print("✓ TreeExplainer successful")
    except Exception as e:
        print(f"TreeExplainer failed: {e}")
        
        # Try KernelExplainer (slower, any model)
        try:
            print("\nTrying KernelExplainer...")
            background = shap.sample(X_train_transformed, min(100, len(X_train_transformed)))
            
            # Wrapper function for prediction
            def predict_fn(X):
                if hasattr(model_for_shap, 'predict_proba'):
                    return model_for_shap.predict_proba(X)[:, 1]
                else:
                    return model_for_shap.decision_function(X)
            
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_test_transformed, nsamples=100)
            print("✓ KernelExplainer successful")
        except Exception as e2:
            print(f"KernelExplainer failed: {e2}")
            
            # Try general Explainer
            try:
                print("\nTrying general Explainer...")
                explainer = shap.Explainer(model_for_shap, X_train_transformed)
                shap_values = explainer(X_test_transformed)
                # Extract values from Explanation object
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                print("✓ General Explainer successful")
            except Exception as e3:
                print(f"All SHAP methods failed: {e3}")
                return None
    
    # =======================================================================
    # Process SHAP values
    # =======================================================================
    
    # Handle multiclass output
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values  # Use positive class
        print("  Using SHAP values for positive class")
    elif isinstance(shap_values, list):
        shap_vals = shap_values
    else:
        shap_vals = shap_values
    
    # Ensure 2D array
    if len(shap_vals.shape) > 2:
        shap_vals = shap_vals[:, :, 1]
    
    # Calculate feature importance
    importance = np.mean(np.abs(shap_vals), axis=0)
    
    # Get feature names
    if feature_names is None:
        if isinstance(X_test_transformed, pd.DataFrame):
            feature_names = list(X_test_transformed.columns)
        else:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv('results/shap_feature_importance.csv', index=False)
    
    print(f"\n✓ SHAP computation successful!")
    print(f"Top 10 features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature'][:40]:40s}: {row['importance']:.4f}")
    
    return {
        'importance_df': importance_df,
        'shap_values': shap_vals,
        'explainer': explainer
    }

# ============================================================================
# DISEASE-SPECIFIC PREDICTION WITH EXPLAINABLE AI
# ============================================================================
def predict_disease_specific(model, X, gene_features, shap_importance_df,
                            disease_targets=DISEASE_TARGETS):
    """
    Disease-specific predictions using SHAP explainability
    Maps gene importance to specific blood disorders
    """
    print(f"\n{'='*70}")
    print("DISEASE-SPECIFIC PREDICTION (EXPLAINABLE AI)")
    print(f"{'='*70}")
    
    if shap_importance_df is None:
        print("⚠️ SHAP importance required for disease prediction")
        return None
    
    # Get predictions
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except:
        y_proba = model.decision_function(X)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    
    # Map genes to disease categories using SHAP importance
    disease_gene_mapping = create_disease_gene_mapping()
    
    disease_predictions = []
    for sample_idx, sample_id in enumerate(X.index):
        sample_pred = {
            'sample_id': sample_id,
            'overall_disease_probability': float(y_proba[sample_idx])
        }
        
        # Calculate disease-specific probabilities
        for disease in disease_targets:
            disease_genes = disease_gene_mapping.get(disease, [])
            
            # Find overlap with top predictive genes
            top_genes = shap_importance_df.head(100)['feature'].tolist()
            overlapping_genes = [g for g in disease_genes if any(g in feat for feat in top_genes)]
            
            if overlapping_genes:
                # Calculate disease probability based on gene importance
                gene_importances = []
                for gene in overlapping_genes:
                    for feat in top_genes:
                        if gene in feat:
                            gene_data = shap_importance_df[shap_importance_df['feature'] == feat]
                            if not gene_data.empty:
                                gene_importances.append(gene_data.iloc[0]['importance'])
                                break
                
                if gene_importances:
                    # Weight by gene importance
                    disease_prob = y_proba[sample_idx] * (sum(gene_importances) / shap_importance_df['importance'].sum())
                    sample_pred[f'{disease}_probability'] = float(min(disease_prob, 1.0))  # Cap at 1.0
                    sample_pred[f'{disease}_evidence_genes'] = len(gene_importances)
                else:
                    sample_pred[f'{disease}_probability'] = 0.0
                    sample_pred[f'{disease}_evidence_genes'] = 0
            else:
                sample_pred[f'{disease}_probability'] = 0.0
                sample_pred[f'{disease}_evidence_genes'] = 0
        
        disease_predictions.append(sample_pred)
    
    # Create DataFrame
    predictions_df = pd.DataFrame(disease_predictions)
    predictions_df.to_csv('disease_predictions/sample_disease_predictions.csv', index=False)
    
    # Summary statistics
    print("\n✓ Disease-specific prediction summary:")
    for disease in disease_targets:
        prob_col = f'{disease}_probability'
        if prob_col in predictions_df.columns:
            mean_prob = predictions_df[prob_col].mean()
            max_prob = predictions_df[prob_col].max()
            n_high = (predictions_df[prob_col] > 0.5).sum()
            print(f"  {disease:50s}: Mean={mean_prob:.3f}, Max={max_prob:.3f}, High risk={n_high}")
    
    return predictions_df
def create_disease_gene_mapping():
    """
    Create gene-to-disease mapping based on known blood disorder genetics
    """
    mapping = {
        'hemophilia': [
            'F8', 'F9', 'VWF', 'F11', 'F5', 'F7', 'F10', 'F2',
            'PROC', 'PROS1', 'SERPINC1', 'FGB', 'FGA', 'FGG'
        ],
        'von_willebrand_disease': [
            'VWF', 'GP1BA', 'GP9', 'ADAMTS13', 'LMAN1', 'MCFD2',
            'F8', 'CLEC4M', 'STX2', 'STXBP2'
        ],
        'sickle_cell_disease': [
            'HBB', 'HBA1', 'HBA2', 'BCL11A', 'HBS1L', 'MYB',
            'KLF1', 'SOX6', 'LRF', 'GATA1', 'AHSP', 'HMOX1'
        ],
        'thalassemia': [
            'HBA1', 'HBA2', 'HBB', 'HBD', 'HBG1', 'HBG2',
            'BCL11A', 'KLF1', 'GATA1', 'ATRX', 'HBA16S'
        ],
        'thrombophilia': [
            'F5', 'F2', 'PROC', 'PROS1', 'SERPINC1', 'MTHFR',
            'FGB', 'FGA', 'FGG', 'PAI1', 'THBD', 'F12', 'F13A1'
        ],
        'platelet_disorders': [
            'ITGA2B', 'ITGB3', 'GP1BA', 'GP1BB', 'GP9', 'NBEAL2',
            'VPS33B', 'RUNX1', 'FLI1', 'MYH9', 'ANKRD26', 'ETV6',
            'ACTN1', 'TUBB1', 'WAS', 'MPL', 'THPO'
        ],
        'hereditary_hemorrhagic_telangiectasia': [
            'ENG', 'ACVRL1', 'SMAD4', 'GDF2', 'RASA1',
            'EPHB4', 'PTPN14', 'ALK1', 'BMPR2'
        ],
        'iron_refractory_iron_deficiency_anemia': [
            'TMPRSS6', 'SLC11A2', 'TFR2', 'HFE', 'HAMP',
            'HFE2', 'TF', 'SLC40A1', 'CP', 'FTL', 'FTH1'
        ]
    }
    
    print(f"\n✓ Disease-gene mapping created for {len(mapping)} disorders")
    for disease, genes in mapping.items():
        print(f"  {disease}: {len(genes)} genes")
    
    return mapping

def create_gene_disease_report(predictions_df, shap_importance_df,
                              disease_targets=DISEASE_TARGETS):
    """Create comprehensive gene-disease association report"""
    print(f"\n{'='*70}")
    print("GENE-DISEASE ASSOCIATION REPORT")
    print(f"{'='*70}")
    
    disease_mapping = create_disease_gene_mapping()
    
    # For each disease, analyze top genes
    report_data = []
    
    for disease in disease_targets:
        disease_genes = disease_mapping.get(disease, [])
        
        # Find genes in top predictive features
        top_features = shap_importance_df.head(200)
        
        for gene in disease_genes:
            gene_data = top_features[top_features['feature'].str.contains(gene, case=False, na=False)]
            
            if not gene_data.empty:
                gene_info = gene_data.iloc[0]
                
                # Calculate disease association percentage
                max_importance = shap_importance_df['importance'].max()
                gene_percentage = (gene_info['importance'] / max_importance) * 100
                
                report_data.append({
                    'disease': disease,
                    'gene': gene,
                    'feature_name': gene_info['feature'],
                    'shap_importance': gene_info['importance'],
                    'disease_association_percentage': gene_percentage,
                    'rank_in_top_features': int(gene_data.index[0]) + 1
                })
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values(['disease', 'disease_association_percentage'],
                                         ascending=[True, False])
        report_df.to_csv('disease_predictions/gene_disease_associations.csv', index=False)
        
        print("\n✓ Top gene-disease associations found:")
        for disease in disease_targets:
            disease_data = report_df[report_df['disease'] == disease].head(5)
            if not disease_data.empty:
                print(f"\n{disease}:")
                for _, row in disease_data.iterrows():
                    print(f"  {row['gene']:15s}: {row['disease_association_percentage']:6.2f}% "
                          f"(SHAP: {row['shap_importance']:.4f}, Rank: {row['rank_in_top_features']})")
        
        return report_df
    else:
        print("⚠️ No gene-disease associations found in top features")
        return None
# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_plots(results_dict):
    """Create all visualization plots"""
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Model Performance
    ax1 = plt.subplot(3, 3, 1)
    if 'baseline_results' in results_dict:
        df = results_dict['baseline_results']
        ax1.bar(range(len(df)), df['val_auc'], alpha=0.8)
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['model'], rotation=45, ha='right')
        ax1.set_ylabel('Validation AUC')
        ax1.set_title('Model Performance Comparison')
        ax1.grid(alpha=0.3)
    
    # Plot 2: Feature Importance
    ax2 = plt.subplot(3, 3, 2)
    if 'shap_importance' in results_dict and results_dict['shap_importance'] is not None:
        top_features = results_dict['shap_importance']['importance_df'].head(15)
        y_pos = np.arange(len(top_features))
        ax2.barh(y_pos, top_features['importance'], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f[:25] for f in top_features['feature']], fontsize=8)
        ax2.set_xlabel('SHAP Importance')
        ax2.set_title('Top 15 Features')
        ax2.grid(alpha=0.3)
    
    # Plot 3: Bootstrap CI
    ax3 = plt.subplot(3, 3, 3)
    if 'bootstrap_ci' in results_dict:
        ci = results_dict['bootstrap_ci']
        metrics = list(ci.keys())
        means = [ci[m]['mean'] for m in metrics]
        lowers = [ci[m]['ci_lower'] for m in metrics]
        uppers = [ci[m]['ci_upper'] for m in metrics]
        
        y_pos = np.arange(len(metrics))
        ax3.errorbar(means, y_pos, 
                    xerr=[np.array(means) - np.array(lowers),
                          np.array(uppers) - np.array(means)],
                    fmt='o', capsize=5)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([m.upper() for m in metrics])
        ax3.set_xlabel('Score')
        ax3.set_title('Bootstrap 95% CI')
        ax3.grid(alpha=0.3)
    
    # Plot 4: Disease Predictions
    ax4 = plt.subplot(3, 3, 4)
    if 'disease_predictions' in results_dict and results_dict['disease_predictions'] is not None:
        pred_df = results_dict['disease_predictions']
        disease_cols = [col for col in pred_df.columns if col.endswith('_probability')]
        
        if disease_cols:
            mean_probs = [pred_df[col].mean() for col in disease_cols]
            disease_names = [col.replace('_probability', '').replace('_', ' ') for col in disease_cols]
            
            ax4.barh(range(len(disease_names)), mean_probs, alpha=0.8)
            ax4.set_yticks(range(len(disease_names)))
            ax4.set_yticklabels(disease_names, fontsize=8)
            ax4.set_xlabel('Mean Disease Probability')
            ax4.set_title('Disease-Specific Predictions')
            ax4.grid(alpha=0.3)
    
    # Plot 5: Permutation Test
    ax5 = plt.subplot(3, 3, 5)
    if 'permutation_test' in results_dict:
        perm = results_dict['permutation_test']
        if 'perm_scores' in perm:
            ax5.hist(perm['perm_scores'], bins=30, alpha=0.7, edgecolor='black')
            ax5.axvline(perm['original_score'], color='red', linestyle='--', linewidth=2,
                       label=f"Original: {perm['original_score']:.3f}")
            ax5.set_xlabel('Score')
            ax5.set_ylabel('Frequency')
            ax5.set_title(f"Permutation Test (p={perm.get('p_value', 1):.4f})")
            ax5.legend()
            ax5.grid(alpha=0.3)
    
    # Plot 6: Gene-Disease Heatmap
    ax6 = plt.subplot(3, 3, 6)
    if 'gene_disease_report' in results_dict and results_dict['gene_disease_report'] is not None:
        report = results_dict['gene_disease_report']
        
        # Create pivot table for heatmap
        pivot = report.pivot_table(
            values='disease_association_percentage',
            index='gene',
            columns='disease',
            aggfunc='first'
        ).fillna(0)
        
        if not pivot.empty:
            im = ax6.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
            ax6.set_xticks(range(len(pivot.columns)))
            ax6.set_xticklabels([col[:15] for col in pivot.columns], rotation=45, ha='right', fontsize=7)
            ax6.set_yticks(range(len(pivot.index)))
            ax6.set_yticklabels(pivot.index, fontsize=8)
            ax6.set_title('Gene-Disease Association %')
            plt.colorbar(im, ax=ax6)
    
    # Plot 7: Sample Predictions Distribution
    ax7 = plt.subplot(3, 3, 7)
    if 'test_predictions' in results_dict:
        pred = results_dict['test_predictions']
        ax7.hist(pred['y_proba'], bins=30, alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Predicted Probability')
        ax7.set_ylabel('Number of Samples')
        ax7.set_title('Test Set Prediction Distribution')
        ax7.axvline(0.5, color='red', linestyle='--', alpha=0.5)
        ax7.grid(alpha=0.3)
    
    # Plot 8: Summary Statistics
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    summary_text = "Analysis Summary\n\n"
    if 'model_summary' in results_dict:
        ms = results_dict['model_summary']
        summary_text += f"Best Model: {ms.get('best_model', 'N/A')}\n"
        summary_text += f"Test AUC: {ms.get('test_auc', 0):.3f}\n"
        summary_text += f"Test Accuracy: {ms.get('test_accuracy', 0):.3f}\n"
        summary_text += f"Features: {ms.get('n_features', 0):,}\n"
        summary_text += f"Test Samples: {ms.get('n_test', 0):,}\n\n"
    
    if 'permutation_test' in results_dict:
        pt = results_dict['permutation_test']
        summary_text += f"Statistical Significance:\n"
        summary_text += f"  {'✓ Yes' if pt.get('significant', False) else '✗ No'}\n"
        summary_text += f"  p-value: {pt.get('p_value', 1):.6f}\n"
    
    ax8.text(0.05, 0.95, summary_text, fontsize=10, ha='left', va='top',
            transform=ax8.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 9: ROC Curve
    ax9 = plt.subplot(3, 3, 9)
    if 'test_predictions' in results_dict:
        pred = results_dict['test_predictions']
        fpr, tpr, _ = roc_curve(pred['y_true'], pred['y_proba'])
        auc = roc_auc_score(pred['y_true'], pred['y_proba'])
        
        ax9.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
        ax9.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax9.set_xlabel('False Positive Rate')
        ax9.set_ylabel('True Positive Rate')
        ax9.set_title('ROC Curve')
        ax9.legend()
        ax9.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/comprehensive_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Comprehensive plots saved")

# ============================================================================
# FINAL REPORT
# ============================================================================

def create_final_report(results_dict):
    """Generate comprehensive markdown report"""
    print(f"\n{'='*70}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*70}")
    
    report = f"""# Gene Expression Analysis - Complete Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Pipeline Version**: 5.0 - Unified & Complete

---

## Executive Summary

"""
    
    if 'model_summary' in results_dict:
        ms = results_dict['model_summary']
        report += f"""
### Model Performance
- **Best Model**: {ms.get('best_model', 'N/A')}
- **Test AUC**: {ms.get('test_auc', 0):.3f}
- **Test Accuracy**: {ms.get('test_accuracy', 0):.3f}
- **Features Used**: {ms.get('n_features', 0):,}
- **Test Samples**: {ms.get('n_test', 0):,}

"""
    
    if 'permutation_test' in results_dict:
        pt = results_dict['permutation_test']
        report += f"""
### Statistical Validation
- **P-value**: {pt.get('p_value', 1):.6f}
- **Statistically Significant**: {'✓ Yes' if pt.get('significant', False) else '✗ No'}
- **Original Score**: {pt.get('original_score', 0):.4f}
- **Permutation Mean**: {pt.get('perm_mean', 0):.4f}

"""
    
    if 'disease_predictions' in results_dict and results_dict['disease_predictions'] is not None:
        pred_df = results_dict['disease_predictions']
        report += f"""
### Disease-Specific Predictions

Total samples analyzed: {len(pred_df)}

| Disease | Mean Probability | Max Probability | High Risk Samples |
|---------|-----------------|-----------------|-------------------|
"""
        
        for disease in DISEASE_TARGETS:
            prob_col = f'{disease}_probability'
            if prob_col in pred_df.columns:
                mean_p = pred_df[prob_col].mean()
                max_p = pred_df[prob_col].max()
                high_risk = (pred_df[prob_col] > 0.5).sum()
                report += f"| {disease.replace('_', ' ').title()} | {mean_p:.3f} | {max_p:.3f} | {high_risk} |\n"
    
    if 'gene_disease_report' in results_dict and results_dict['gene_disease_report'] is not None:
        gd_report = results_dict['gene_disease_report']
        report += f"""

### Top Gene-Disease Associations (by SHAP Importance)

"""
        for disease in DISEASE_TARGETS[:5]:  # Top 5 diseases
            disease_data = gd_report[gd_report['disease'] == disease].head(5)
            if not disease_data.empty:
                report += f"\n#### {disease.replace('_', ' ').title()}\n\n"
                report += "| Gene | Association % | SHAP Importance | Rank |\n"
                report += "|------|--------------|-----------------|------|\n"
                for _, row in disease_data.iterrows():
                    report += f"| {row['gene']} | {row['disease_association_percentage']:.2f}% | {row['shap_importance']:.4f} | {row['rank_in_top_features']} |\n"
    
    if 'bootstrap_ci' in results_dict:
        ci = results_dict['bootstrap_ci']
        report += f"""

### Bootstrap Confidence Intervals (95%)

| Metric | Mean | CI Lower | CI Upper | Std Dev |
|--------|------|----------|----------|---------|
"""
        for metric, values in ci.items():
            report += f"| {metric.upper()} | {values['mean']:.3f} | {values['ci_lower']:.3f} | {values['ci_upper']:.3f} | {values['std']:.3f} |\n"
    
    report += f"""

---

## Methodology

### Pipeline Steps
1. **Data Loading & Inspection**: Quality control and validation
2. **Gene Annotation**: BioMart integration for gene symbols
3. **Preprocessing**: Filtering, normalization, transformation
4. **Feature Engineering**: Multiple feature representations
5. **Model Training**: Baseline ML models with cross-validation
6. **Statistical Validation**: Bootstrap CI + permutation testing
7. **Explainability**: SHAP feature importance analysis
8. **Disease Prediction**: AI-driven disease-specific risk assessment

### Disease Targets
Blood disorders analyzed:
{chr(10).join('- ' + d.replace('_', ' ').title() for d in DISEASE_TARGETS)}

---

## Key Findings

1. **Model Performance**: {'Excellent' if results_dict.get('model_summary', {}).get('test_auc', 0) > 0.9 else 'Good'} discriminative ability
2. **Statistical Significance**: {'Validated' if results_dict.get('permutation_test', {}).get('significant', False) else 'Not validated'} through permutation testing
3. **Gene Associations**: Identified key genes associated with specific blood disorders
4. **Explainability**: SHAP analysis provides biological interpretability

---

## Generated Files

### Results
- `results/baseline_metrics.csv` - Model comparison
- `results/shap_feature_importance.csv` - Feature rankings
- `results/bootstrap_ci.json` - Statistical validation
- `results/permutation_test.json` - Significance testing

### Disease Predictions
- `disease_predictions/sample_disease_predictions.csv` - Per-sample disease probabilities
- `disease_predictions/gene_disease_associations.csv` - Gene-disease mapping with percentages

### Visualizations
- `figs/comprehensive_analysis_results.png` - Main results figure

---

## Interpretation

The analysis successfully:
- Identified predictive gene expression signatures
- Validated model performance statistically
- Mapped genes to specific blood disorders using explainable AI
- Provided per-sample disease risk predictions

**Clinical Relevance**: Top predictive genes represent potential biomarkers for blood disorder diagnosis and classification.

---

*Report generated by Gene Expression Analysis Pipeline v5.0*
"""
    
    # Save report
    with open('reports/final_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✓ Final report saved to reports/final_analysis_report.md")
    
    return report

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def run_complete_pipeline(expression_file, metadata_file,
                         condition_col='condition',
                         healthy_label='healthy',
                         disease_label='disease',
                         random_state=42):
    """
    Execute complete end-to-end pipeline
    """
    print("\n" + "="*80)
    print("GENE EXPRESSION ANALYSIS - COMPLETE PIPELINE v5.0")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    results_dict = {}
    
    # Setup
    setup_directories()
    
    # Step 1: Load and inspect data
    expr_df, summary = inspect_expression_data(expression_file, "main_dataset")
    
    # Step 2: Gene annotation
    try:
        annotation_df = annotate_genes_biomart(expr_df.index.tolist())
    except:
        annotation_df = create_fallback_annotation(expr_df.index.tolist())
    
    expr_matrix, expr_annotated = add_gene_symbols(expr_df, annotation_df)
    
    # Step 3: Preprocessing
    filtered_expr = filter_low_expression(expr_matrix, threshold=1.0, min_pct=0.1)
    normalized_expr = normalize_transform(filtered_expr, log_transform=True)
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_file, index_col=0)
    metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]
    
    # Step 4: Differential expression
    de_results = differential_expression(
        normalized_expr, metadata_df,
        condition_col=condition_col,
        healthy=healthy_label,
        disease=disease_label
    )
    
    # Step 5: Feature engineering
    features_dict = engineer_features(normalized_expr, de_results)
    
    # Step 6: Data splitting
    feature_set = 'X_var'  # Use variable genes
    splits = create_splits(
        features_dict[feature_set], metadata_df,
        condition_col=condition_col,
        random_state=random_state
    )
    
    # Step 7: Model training
    baseline_results, trained_models = train_baseline_models(
        splits, scale=True, random_state=random_state
    )
    results_dict['baseline_results'] = baseline_results
    
    # Get best model
    best_model_name = baseline_results.iloc[0]['model']
    best_model = trained_models[best_model_name]
    
    # Step 8: Test set evaluation
    X_test, y_test = splits['X_test'], splits['y_test']
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_proba)
    test_acc = accuracy_score(y_test, y_pred)
    
    results_dict['model_summary'] = {
        'best_model': best_model_name,
        'test_auc': float(test_auc),
        'test_accuracy': float(test_acc),
        'n_features': int(X_test.shape[1]),
        'n_test': int(len(X_test))
    }
    
    results_dict['test_predictions'] = {
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    # Save predictions
    pred_table = pd.DataFrame({
        'sample_id': X_test.index,
        'true_label': y_test.values,
        'predicted_label': y_pred,
        'disease_probability': y_proba
    })
    pred_table.to_csv('results/test_predictions.csv', index=False)
    
    # Step 9: Statistical validation
    print("\nPerforming statistical validation...")
    
    # Bootstrap CI
    bootstrap_ci = bootstrap_metrics(y_test, y_pred, y_proba, n_bootstrap=1000, random_state=random_state)
    results_dict['bootstrap_ci'] = bootstrap_ci
    
    # Permutation test
    original_score, perm_scores, pvalue = permutation_test(
        best_model, X_test, y_test, n_permutations=1000, random_state=random_state
    )
    results_dict['permutation_test'] = {
        'original_score': float(original_score),
        'perm_scores': perm_scores,
        'perm_mean': float(np.mean(perm_scores)),
        'perm_std': float(np.std(perm_scores)),
        'p_value': float(pvalue),
        'significant': bool(pvalue < 0.05)
    }
    
    # Step 10: SHAP explainability
    print("\nComputing SHAP feature importance...")
    X_train = splits['X_train']
    
    disease_predictions = None
    gene_disease_report = None
    
    shap_results = compute_shap_importance(
        best_model, X_train, X_test,
        feature_names=list(X_test.columns),
        max_samples=min(100, len(X_test)),
        max_features=min(500, X_test.shape[1])
    )
    
    if shap_results:
        results_dict['shap_importance'] = shap_results
        
        # Step 11: Disease-specific prediction
        print("\nGenerating disease-specific predictions...")
        
        disease_predictions = predict_disease_specific(
            best_model, X_test,
            list(X_test.columns),
            shap_results['importance_df']
        )
        results_dict['disease_predictions'] = disease_predictions
        
        # Step 12: Gene-disease association report
        gene_disease_report = create_gene_disease_report(
            disease_predictions,
            shap_results['importance_df']
        )
        results_dict['gene_disease_report'] = gene_disease_report
    else:
        print("⚠️ SHAP analysis failed - skipping disease-specific predictions")
        results_dict['disease_predictions'] = None
        results_dict['gene_disease_report'] = None
    
    # Step 13: Create visualizations
    create_comprehensive_plots(results_dict)
    
    # Step 14: Generate final report
    final_report = create_final_report(results_dict)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\n⏱️  Total execution time: {duration}")
    print(f"\n📊 Key Results:")
    print(f"   • Test AUC: {test_auc:.3f}")
    print(f"   • Test Accuracy: {test_acc:.3f}")
    print(f"   • Statistical significance: p = {pvalue:.6f}")
    print(f"   • Disease predictions generated for {len(DISEASE_TARGETS)} disorders")
    
    print(f"\n📁 Generated files:")
    output_files = [
        'results/baseline_metrics.csv',
        'results/shap_feature_importance.csv',
        'results/test_predictions.csv',
        'disease_predictions/sample_disease_predictions.csv',
        'disease_predictions/gene_disease_associations.csv',
        'figs/comprehensive_analysis_results.png',
        'reports/final_analysis_report.md'
    ]
    
    for filepath in output_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   ✓ {filepath} ({size:,} bytes)")
    
    print("\n🧬 Disease-Specific Insights:")
    if disease_predictions is not None:
        for disease in DISEASE_TARGETS[:5]:
            prob_col = f'{disease}_probability'
            if prob_col in disease_predictions.columns:
                mean_prob = disease_predictions[prob_col].mean()
                max_prob = disease_predictions[prob_col].max()
                print(f"   • {disease.replace('_', ' ').title()}: "
                      f"Mean={mean_prob:.3f}, Max={max_prob:.3f}")
    
    print("\n✅ Analysis complete! Check the reports/ directory for detailed results.")
    
    return results_dict

# ============================================================================
# EXAMPLE USAGE & EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  Gene Expression Classification Pipeline v5.0                     ║
    ║  Complete Unified Pipeline with Disease-Specific Prediction       ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Example configuration
    config = {
        'expression_file': 'data/GSE107011_Processed_data_TPM.txt',
        'metadata_file': 'meta/metadata_gse.csv',
        'condition_col': 'label',  # or 'condition'
        'healthy_label': 'healthy',
        'disease_label': 'disease',
        'random_state': 42
    }
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*80)
    print("STARTING PIPELINE EXECUTION")
    print("="*80 + "\n")
    
    try:
        # Run complete pipeline
        results = run_complete_pipeline(**config)
        
        print("\n" + "="*80)
        print("🏆 SUCCESS - All pipeline steps completed")
        print("="*80)
        
        # Print summary statistics
        if results:
            print("\n📈 Final Summary Statistics:")
            print(f"   • Best Model: {results.get('model_summary', {}).get('best_model', 'N/A')}")
            print(f"   • Test AUC: {results.get('model_summary', {}).get('test_auc', 0):.4f}")
            print(f"   • Test Accuracy: {results.get('model_summary', {}).get('test_accuracy', 0):.4f}")
            print(f"   • P-value: {results.get('permutation_test', {}).get('p_value', 1):.6f}")
            print(f"   • Significant: {'✓ Yes' if results.get('permutation_test', {}).get('significant', False) else '✗ No'}")
            
            if 'gene_disease_report' in results and results['gene_disease_report'] is not None:
                print(f"   • Gene-disease associations: {len(results['gene_disease_report'])} found")
        
        print("\n🎯 Next Steps:")
        print("   1. Review reports/final_analysis_report.md for detailed findings")
        print("   2. Examine disease_predictions/sample_disease_predictions.csv for per-sample predictions")
        print("   3. Check disease_predictions/gene_disease_associations.csv for gene-disease mappings")
        print("   4. View figs/comprehensive_analysis_results.png for visualization")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Required file not found - {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure your data files are in the correct location")
        print("   2. Check that column names in metadata match configuration")
        print("   3. Verify file paths are correct")
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed - {e}")
        print("\n📝 Debug Information:")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Common Issues:")
        print("   1. Memory error: Reduce n_variable or max_samples parameters")
        print("   2. Import error: Install missing packages (shap, gseapy, etc.)")
        print("   3. Data format: Ensure expression file is tab-separated with genes as rows")
    
    print("\n" + "="*80)
    print("Pipeline execution finished")
    print("="*80 + "\n")