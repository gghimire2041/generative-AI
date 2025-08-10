# Comprehensive Error Analysis for 8-Class Neural Network Classification Model
# This notebook provides detailed error analysis with beautiful visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            precision_recall_curve, roc_curve, auc,
                            precision_score, recall_score, f1_score,
                            matthews_corrcoef, cohen_kappa_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load your data
# Replace 'your_file.csv' with your actual file path
# df = pd.read_csv('your_file.csv')

# For demonstration, creating sample data structure
# Replace this section with your actual data loading
np.random.seed(42)
n_samples = 1000

# Define class names based on your data
class_names = ['access_issue', 'courier', 'hardware', 'manufacturer', 
               'software', 'telecom', 'user_guidance', 'other']

# Sample data creation (replace with your actual data)
sample_data = {
    'INC_Number': [f'INC{i:06d}' for i in range(n_samples)],
    'Real_Label': np.random.choice(class_names, n_samples),
    'Predicted_Label': np.random.choice(class_names, n_samples),
}

# Add probability columns
for class_name in class_names:
    prob_col = f'Prob_{class_name}'
    sample_data[prob_col] = np.random.random(n_samples)

# Normalize probabilities to sum to 1 for each row
prob_cols = [f'Prob_{name}' for name in class_names]
prob_matrix = np.array([sample_data[col] for col in prob_cols]).T
prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
for i, col in enumerate(prob_cols):
    sample_data[col] = prob_matrix[:, i]

df = pd.DataFrame(sample_data)

print("🔍 COMPREHENSIVE ERROR ANALYSIS FOR 8-CLASS CLASSIFICATION MODEL")
print("=" * 70)
print(f"Dataset shape: {df.shape}")
print(f"Classes: {class_names}")
print("\nFirst few rows:")
print(df.head())

# ============================================================================
# 1. BASIC PERFORMANCE METRICS OVERVIEW
# ============================================================================

print("\n📊 1. BASIC PERFORMANCE METRICS")
print("-" * 50)

# Get actual and predicted labels
y_true = df['Real_Label']
y_pred = df['Predicted_Label']

# Calculate basic metrics
accuracy = (y_true == y_pred).mean()
print(f"Overall Accuracy: {accuracy:.4f}")

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Additional metrics
macro_precision = precision_score(y_true, y_pred, average='macro')
macro_recall = recall_score(y_true, y_pred, average='macro')
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

print(f"\nAdditional Metrics:")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")

# ============================================================================
# 2. CONFUSION MATRIX ANALYSIS
# ============================================================================

def plot_confusion_matrices():
    """Create comprehensive confusion matrix visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Raw counts confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0,0], cbar_kws={'label': 'Count'})
    axes[0,0].set_title('Confusion Matrix - Raw Counts', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Predicted Label')
    axes[0,0].set_ylabel('True Label')
    
    # 2. Normalized confusion matrix (by true class)
    cm_norm = confusion_matrix(y_true, y_pred, labels=class_names, normalize='true')
    
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0,1], cbar_kws={'label': 'Recall'})
    axes[0,1].set_title('Normalized Confusion Matrix - Recall', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Predicted Label')
    axes[0,1].set_ylabel('True Label')
    
    # 3. Normalized confusion matrix (by predicted class)
    cm_norm_pred = confusion_matrix(y_true, y_pred, labels=class_names, normalize='pred')
    
    sns.heatmap(cm_norm_pred, annot=True, fmt='.3f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1,0], cbar_kws={'label': 'Precision'})
    axes[1,0].set_title('Normalized Confusion Matrix - Precision', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Predicted Label')
    axes[1,0].set_ylabel('True Label')
    
    # 4. Error matrix (showing only misclassifications)
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    
    sns.heatmap(cm_errors, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1,1], cbar_kws={'label': 'Error Count'})
    axes[1,1].set_title('Error Matrix - Misclassifications Only', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Predicted Label')
    axes[1,1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    return cm, cm_norm, cm_norm_pred

print("\n🎯 2. CONFUSION MATRIX ANALYSIS")
print("-" * 50)
cm, cm_norm, cm_norm_pred = plot_confusion_matrices()

# ============================================================================
# 3. CLASS-WISE PERFORMANCE ANALYSIS
# ============================================================================

def analyze_class_performance():
    """Detailed class-wise performance analysis"""
    
    # Calculate per-class metrics
    class_metrics = []
    
    for i, class_name in enumerate(class_names):
        # Get metrics for this class
        class_precision = precision_score(y_true, y_pred, labels=[class_name], average=None)[0] if class_name in y_pred.values else 0
        class_recall = recall_score(y_true, y_pred, labels=[class_name], average=None)[0] if class_name in y_true.values else 0
        class_f1 = f1_score(y_true, y_pred, labels=[class_name], average=None)[0] if class_name in y_true.values and class_name in y_pred.values else 0
        
        # Support (number of true instances)
        support = sum(y_true == class_name)
        
        # Predicted instances
        predicted_count = sum(y_pred == class_name)
        
        class_metrics.append({
            'Class': class_name,
            'Precision': class_precision,
            'Recall': class_recall,
            'F1-Score': class_f1,
            'Support': support,
            'Predicted_Count': predicted_count
        })
    
    metrics_df = pd.DataFrame(class_metrics)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Precision, Recall, F1 comparison
    x_pos = np.arange(len(class_names))
    width = 0.25
    
    axes[0,0].bar(x_pos - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    axes[0,0].bar(x_pos, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    axes[0,0].bar(x_pos + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    axes[0,0].set_xlabel('Classes')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_title('Class-wise Performance Metrics', fontweight='bold')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(class_names, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Support vs Predicted Count
    axes[0,1].bar(x_pos - width/2, metrics_df['Support'], width, label='True Count', alpha=0.8)
    axes[0,1].bar(x_pos + width/2, metrics_df['Predicted_Count'], width, label='Predicted Count', alpha=0.8)
    
    axes[0,1].set_xlabel('Classes')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('True vs Predicted Class Distribution', fontweight='bold')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(class_names, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Class imbalance visualization
    class_distribution = y_true.value_counts()
    axes[1,0].pie(class_distribution.values, labels=class_distribution.index, autopct='%1.1f%%')
    axes[1,0].set_title('True Class Distribution', fontweight='bold')
    
    # 4. Prediction vs Reality scatter
    pred_distribution = y_pred.value_counts().reindex(class_names, fill_value=0)
    true_distribution = y_true.value_counts().reindex(class_names, fill_value=0)
    
    axes[1,1].scatter(true_distribution.values, pred_distribution.values, s=100, alpha=0.7)
    for i, class_name in enumerate(class_names):
        axes[1,1].annotate(class_name, (true_distribution[class_name], pred_distribution[class_name]),
                          xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal line for perfect prediction
    max_val = max(max(true_distribution.values), max(pred_distribution.values))
    axes[1,1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    axes[1,1].set_xlabel('True Count')
    axes[1,1].set_ylabel('Predicted Count')
    axes[1,1].set_title('Prediction vs Reality Distribution', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df

print("\n📈 3. CLASS-WISE PERFORMANCE ANALYSIS")
print("-" * 50)
metrics_df = analyze_class_performance()
print("\nClass-wise Metrics Summary:")
print(metrics_df.round(4))

# ============================================================================
# 4. CONFIDENCE ANALYSIS
# ============================================================================

def analyze_prediction_confidence():
    """Analyze model confidence and calibration"""
    
    # Get probability columns
    prob_cols = [f'Prob_{name}' for name in class_names]
    
    # Calculate max probability for each prediction
    max_probs = df[prob_cols].max(axis=1)
    predicted_classes = df[prob_cols].idxmax(axis=1).str.replace('Prob_', '')
    
    # Calculate confidence statistics
    df['Max_Probability'] = max_probs
    df['Predicted_Class_Confidence'] = predicted_classes
    df['Is_Correct'] = (df['Real_Label'] == df['Predicted_Label'])
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Confidence distribution for correct vs incorrect predictions
    correct_conf = df[df['Is_Correct']]['Max_Probability']
    incorrect_conf = df[~df['Is_Correct']]['Max_Probability']
    
    axes[0,0].hist(correct_conf, bins=30, alpha=0.7, label='Correct', density=True)
    axes[0,0].hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', density=True)
    axes[0,0].set_xlabel('Max Probability')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Calibration plot
    # Bin predictions by confidence
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = df[in_bin]['Is_Correct'].mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
    
    axes[0,1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    axes[0,1].plot(confidences, accuracies, 'bo-', label='Model Calibration')
    axes[0,1].set_xlabel('Mean Predicted Probability')
    axes[0,1].set_ylabel('Fraction of Positives')
    axes[0,1].set_title('Calibration Plot', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Confidence vs accuracy by bins
    confidence_bins = pd.cut(max_probs, bins=10)
    conf_analysis = df.groupby(confidence_bins).agg({
        'Is_Correct': ['mean', 'count'],
        'Max_Probability': 'mean'
    }).round(3)
    
    bin_centers = [interval.mid for interval in conf_analysis.index]
    accuracies_by_bin = conf_analysis[('Is_Correct', 'mean')]
    counts_by_bin = conf_analysis[('Is_Correct', 'count')]
    
    bars = axes[1,0].bar(range(len(bin_centers)), accuracies_by_bin, alpha=0.7)
    axes[1,0].set_xlabel('Confidence Bins')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_title('Accuracy by Confidence Bins', fontweight='bold')
    axes[1,0].set_xticks(range(len(bin_centers)))
    axes[1,0].set_xticklabels([f'{x:.2f}' for x in bin_centers], rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, counts_by_bin)):
        axes[1,0].annotate(f'n={count}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                          ha='center', va='bottom')
    
    # 4. Class-wise confidence analysis
    class_conf_stats = []
    for class_name in class_names:
        class_mask = df['Real_Label'] == class_name
        if class_mask.sum() > 0:
            correct_mask = class_mask & df['Is_Correct']
            incorrect_mask = class_mask & ~df['Is_Correct']
            
            avg_conf_correct = df[correct_mask]['Max_Probability'].mean() if correct_mask.sum() > 0 else 0
            avg_conf_incorrect = df[incorrect_mask]['Max_Probability'].mean() if incorrect_mask.sum() > 0 else 0
            
            class_conf_stats.append({
                'Class': class_name,
                'Avg_Conf_Correct': avg_conf_correct,
                'Avg_Conf_Incorrect': avg_conf_incorrect,
                'Confidence_Gap': avg_conf_correct - avg_conf_incorrect
            })
    
    conf_stats_df = pd.DataFrame(class_conf_stats)
    
    x_pos = np.arange(len(class_names))
    width = 0.35
    
    axes[1,1].bar(x_pos - width/2, conf_stats_df['Avg_Conf_Correct'], width, 
                  label='Correct Predictions', alpha=0.8)
    axes[1,1].bar(x_pos + width/2, conf_stats_df['Avg_Conf_Incorrect'], width,
                  label='Incorrect Predictions', alpha=0.8)
    
    axes[1,1].set_xlabel('Classes')
    axes[1,1].set_ylabel('Average Confidence')
    axes[1,1].set_title('Average Confidence by Class and Correctness', fontweight='bold')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(class_names, rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df, conf_stats_df

print("\n🎯 4. CONFIDENCE ANALYSIS")
print("-" * 50)
df_with_conf, conf_stats_df = analyze_prediction_confidence()
print("\nConfidence Statistics by Class:")
print(conf_stats_df.round(4))

# ============================================================================
# 5. ERROR PATTERN ANALYSIS
# ============================================================================

def analyze_error_patterns():
    """Deep dive into error patterns and misclassification analysis"""
    
    # Get misclassified samples
    misclassified = df_with_conf[~df_with_conf['Is_Correct']].copy()
    
    print(f"Total misclassifications: {len(misclassified)} out of {len(df_with_conf)} ({len(misclassified)/len(df_with_conf)*100:.2f}%)")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Most common misclassification pairs
    error_pairs = misclassified.groupby(['Real_Label', 'Predicted_Label']).size().reset_index(name='Count')
    error_pairs = error_pairs.sort_values('Count', ascending=False).head(15)
    
    error_labels = [f"{row['Real_Label']} → {row['Predicted_Label']}" for _, row in error_pairs.iterrows()]
    
    axes[0,0].barh(range(len(error_pairs)), error_pairs['Count'])
    axes[0,0].set_yticks(range(len(error_pairs)))
    axes[0,0].set_yticklabels(error_labels)
    axes[0,0].set_xlabel('Error Count')
    axes[0,0].set_title('Top 15 Misclassification Patterns', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Error rate by confidence level
    conf_bins = pd.cut(misclassified['Max_Probability'], bins=10)
    error_by_conf = misclassified.groupby(conf_bins).size()
    total_by_conf = df_with_conf.groupby(pd.cut(df_with_conf['Max_Probability'], bins=10)).size()
    error_rate_by_conf = (error_by_conf / total_by_conf).fillna(0)
    
    bin_centers = [interval.mid for interval in error_rate_by_conf.index]
    
    axes[0,1].plot(bin_centers, error_rate_by_conf.values, 'ro-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Confidence Level')
    axes[0,1].set_ylabel('Error Rate')
    axes[0,1].set_title('Error Rate by Confidence Level', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Class-wise error analysis
    class_errors = []
    for class_name in class_names:
        true_class_mask = df_with_conf['Real_Label'] == class_name
        pred_class_mask = df_with_conf['Predicted_Label'] == class_name
        
        # False Negatives (missed detections)
        fn_mask = true_class_mask & ~df_with_conf['Is_Correct']
        fn_count = fn_mask.sum()
        
        # False Positives (false alarms)
        fp_mask = pred_class_mask & ~df_with_conf['Is_Correct']
        fp_count = fp_mask.sum()
        
        # True Positives
        tp_mask = true_class_mask & df_with_conf['Is_Correct']
        tp_count = tp_mask.sum()
        
        class_errors.append({
            'Class': class_name,
            'False_Negatives': fn_count,
            'False_Positives': fp_count,
            'True_Positives': tp_count
        })
    
    error_df = pd.DataFrame(class_errors)
    
    x_pos = np.arange(len(class_names))
    width = 0.35
    
    axes[1,0].bar(x_pos - width/2, error_df['False_Negatives'], width, 
                  label='False Negatives (Missed)', alpha=0.8, color='red')
    axes[1,0].bar(x_pos + width/2, error_df['False_Positives'], width,
                  label='False Positives (False Alarms)', alpha=0.8, color='orange')
    
    axes[1,0].set_xlabel('Classes')
    axes[1,0].set_ylabel('Error Count')
    axes[1,0].set_title('False Negatives vs False Positives by Class', fontweight='bold')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(class_names, rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Confusion between similar classes (heatmap of normalized errors)
    cm_errors = confusion_matrix(y_true, y_pred, labels=class_names)
    np.fill_diagonal(cm_errors, 0)  # Remove diagonal (correct predictions)
    
    # Normalize by row (true class) to show where each class gets confused
    cm_errors_norm = cm_errors / cm_errors.sum(axis=1, keepdims=True)
    cm_errors_norm = np.nan_to_num(cm_errors_norm)  # Handle division by zero
    
    sns.heatmap(cm_errors_norm, annot=True, fmt='.3f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1,1], cbar_kws={'label': 'Error Rate'})
    axes[1,1].set_title('Class Confusion Patterns (Normalized)', fontweight='bold')
    axes[1,1].set_xlabel('Predicted Label')
    axes[1,1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    return error_pairs, error_df

print("\n⚠️ 5. ERROR PATTERN ANALYSIS")
print("-" * 50)
error_pairs, error_df = analyze_error_patterns()
print("\nTop Error Patterns:")
print(error_pairs.head(10))

# ============================================================================
# 6. ROC AND PRECISION-RECALL CURVES (One-vs-Rest)
# ============================================================================

def plot_roc_and_pr_curves():
    """Plot ROC and Precision-Recall curves for each class"""
    
    # Binarize the labels for one-vs-rest analysis
    prob_cols = [f'Prob_{name}' for name in class_names]
    y_proba = df[prob_cols].values
    
    # Create binary labels for each class
    y_bin = label_binarize(y_true, classes=class_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Colors for each class
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # 1. ROC Curves
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        axes[0,0].plot(fpr, tpr, color=color, lw=2,
                       label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    axes[0,0].plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    axes[0,0].set_xlim([0.0, 1.0])
    axes[0,0].set_ylim([0.0, 1.05])
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curves (One-vs-Rest)', fontweight='bold')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        
        axes[0,1].plot(recall, precision, color=color, lw=2,
                       label=f'{class_name} (AUC = {pr_auc:.3f})')
    
    # Add baseline (random classifier)
    baseline = y_bin.mean(axis=0)
    for i, (class_name, bl) in enumerate(zip(class_names, baseline)):
        axes[0,1].axhline(y=bl, color=colors[i], linestyle='--', alpha=0.5)
    
    axes[0,1].set_xlim([0.0, 1.0])
    axes[0,1].set_ylim([0.0, 1.05])
    axes[0,1].set_xlabel('Recall')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].set_title('Precision-Recall Curves (One-vs-Rest)', fontweight='bold')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. AUC Comparison
    roc_aucs = []
    pr_aucs = []
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
    
    x_pos = np.arange(len(class_names))
    width = 0.35
    
    axes[1,0].bar(x_pos - width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)
    axes[1,0].bar(x_pos + width/2, pr_aucs, width, label='PR-AUC', alpha=0.8)
    
    axes[1,0].set_xlabel('Classes')
    axes[1,0].set_ylabel('AUC Score')
    axes[1,0].set_title('AUC Comparison: ROC vs Precision-Recall', fontweight='bold')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(class_names, rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim([0, 1])
    
    # 4. Threshold Analysis for Best F1 Score
    f1_scores_by_class = []
    best_thresholds = []
    
    for i, class_name in enumerate(class_names):
        precision, recall, thresholds = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        
        best_threshold_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_threshold_idx]
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
        f1_scores_by_class.append(best_f1)
        best_thresholds.append(best_threshold)
    
    # Create scatter plot
    scatter = axes[1,1].scatter(best_thresholds, f1_scores_by_class, 
                               c=range(len(class_names)), s=100, alpha=0.7, cmap='viridis')
    
    for i, class_name in enumerate(class_names):
        axes[1,1].annotate(class_name, (best_thresholds[i], f1_scores_by_class[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    axes[1,1].set_xlabel('Optimal Threshold')
    axes[1,1].set_ylabel('Best F1 Score')
    axes[1,1].set_title('Optimal Thresholds for Maximum F1 Score', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create summary table
    auc_summary = pd.DataFrame({
        'Class': class_names,
        'ROC_AUC': roc_aucs,
        'PR_AUC': pr_aucs,
        'Best_F1': f1_scores_by_class,
        'Optimal_Threshold': best_thresholds
    })
    
    return auc_summary

print("\n📊 6. ROC AND PRECISION-RECALL CURVES")
print("-" * 50)
auc_summary = plot_roc_and_pr_curves()
print("\nAUC Summary:")
print(auc_summary.round(4))

# ============================================================================
# 7. PROBABILITY DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_probability_distributions():
    """Analyze the distribution of predicted probabilities"""
    
    prob_cols = [f'Prob_{name}' for name in class_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Probability distributions for each class
    for i, class_name in enumerate(class_names):
        prob_col = f'Prob_{class_name}'
        
        # Separate by correct vs incorrect predictions
        correct_mask = (df_with_conf['Real_Label'] == class_name) & df_with_conf['Is_Correct']
        incorrect_mask = (df_with_conf['Real_Label'] == class_name) & ~df_with_conf['Is_Correct']
        
        if correct_mask.sum() > 0:
            axes[0,0].hist(df_with_conf[correct_mask][prob_col], bins=20, alpha=0.5, 
                          label=f'{class_name} (Correct)', density=True)
        if incorrect_mask.sum() > 0:
            axes[0,0].hist(df_with_conf[incorrect_mask][prob_col], bins=20, alpha=0.5,
                          label=f'{class_name} (Incorrect)', density=True)
    
    axes[0,0].set_xlabel('Predicted Probability')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Probability Distributions: Correct vs Incorrect', fontweight='bold')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Entropy analysis
    def calculate_entropy(probs):
        """Calculate Shannon entropy"""
        probs = np.array(probs)
        probs = probs + 1e-15  # Add small value to avoid log(0)
        return -np.sum(probs * np.log2(probs), axis=1)
    
    entropy = calculate_entropy(df_with_conf[prob_cols].values)
    df_with_conf['Entropy'] = entropy
    
    # Plot entropy distribution
    correct_entropy = df_with_conf[df_with_conf['Is_Correct']]['Entropy']
    incorrect_entropy = df_with_conf[~df_with_conf['Is_Correct']]['Entropy']
    
    axes[0,1].hist(correct_entropy, bins=30, alpha=0.7, label='Correct', density=True)
    axes[0,1].hist(incorrect_entropy, bins=30, alpha=0.7, label='Incorrect', density=True)
    axes[0,1].set_xlabel('Prediction Entropy')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Prediction Uncertainty (Entropy) Distribution', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Top-2 probability analysis
    # Get top 2 probabilities for each prediction
    top2_probs = np.sort(df_with_conf[prob_cols].values, axis=1)[:, -2:]
    prob_gap = top2_probs[:, 1] - top2_probs[:, 0]  # Difference between top 2
    
    df_with_conf['Prob_Gap'] = prob_gap
    
    correct_gap = df_with_conf[df_with_conf['Is_Correct']]['Prob_Gap']
    incorrect_gap = df_with_conf[~df_with_conf['Is_Correct']]['Prob_Gap']
    
    axes[1,0].hist(correct_gap, bins=30, alpha=0.7, label='Correct', density=True)
    axes[1,0].hist(incorrect_gap, bins=30, alpha=0.7, label='Incorrect', density=True)
    axes[1,0].set_xlabel('Probability Gap (Top1 - Top2)')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Decision Margin Analysis', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Class-wise probability statistics
    prob_stats = []
    for class_name in class_names:
        prob_col = f'Prob_{class_name}'
        class_mask = df_with_conf['Real_Label'] == class_name
        
        if class_mask.sum() > 0:
            correct_probs = df_with_conf[class_mask & df_with_conf['Is_Correct']][prob_col]
            incorrect_probs = df_with_conf[class_mask & ~df_with_conf['Is_Correct']][prob_col]
            
            prob_stats.append({
                'Class': class_name,
                'Mean_Prob_Correct': correct_probs.mean() if len(correct_probs) > 0 else 0,
                'Mean_Prob_Incorrect': incorrect_probs.mean() if len(incorrect_probs) > 0 else 0,
                'Std_Prob_Correct': correct_probs.std() if len(correct_probs) > 0 else 0,
                'Std_Prob_Incorrect': incorrect_probs.std() if len(incorrect_probs) > 0 else 0
            })
    
    prob_stats_df = pd.DataFrame(prob_stats)
    
    x_pos = np.arange(len(class_names))
    width = 0.35
    
    bars1 = axes[1,1].bar(x_pos - width/2, prob_stats_df['Mean_Prob_Correct'], width,
                         yerr=prob_stats_df['Std_Prob_Correct'], label='Correct', alpha=0.8,
                         capsize=5)
    bars2 = axes[1,1].bar(x_pos + width/2, prob_stats_df['Mean_Prob_Incorrect'], width,
                         yerr=prob_stats_df['Std_Prob_Incorrect'], label='Incorrect', alpha=0.8,
                         capsize=5)
    
    axes[1,1].set_xlabel('Classes')
    axes[1,1].set_ylabel('Mean Probability')
    axes[1,1].set_title('Mean Predicted Probability by Class and Correctness', fontweight='bold')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(class_names, rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return prob_stats_df

print("\n🎲 7. PROBABILITY DISTRIBUTION ANALYSIS")
print("-" * 50)
prob_stats_df = analyze_probability_distributions()
print("\nProbability Statistics by Class:")
print(prob_stats_df.round(4))

# ============================================================================
# 8. MISCLASSIFICATION DEEP DIVE
# ============================================================================

def misclassification_deep_dive():
    """Detailed analysis of misclassified samples"""
    
    misclassified = df_with_conf[~df_with_conf['Is_Correct']].copy()
    
    print(f"\n📋 MISCLASSIFICATION SAMPLE ANALYSIS")
    print(f"Analyzing {len(misclassified)} misclassified samples...")
    
    # Find most confident wrong predictions
    top_confident_wrong = misclassified.nlargest(10, 'Max_Probability')
    
    print("\n🚨 TOP 10 MOST CONFIDENT WRONG PREDICTIONS:")
    print("-" * 60)
    for idx, row in top_confident_wrong.iterrows():
        print(f"ID: {row['INC_Number']}")
        print(f"  True: {row['Real_Label']} → Predicted: {row['Predicted_Label']}")
        print(f"  Confidence: {row['Max_Probability']:.4f}")
        print(f"  True class probability: {row[f'Prob_{row[\"Real_Label\"]}']:.4f}")
        print()
    
    # Find least confident correct predictions
    correct_preds = df_with_conf[df_with_conf['Is_Correct']].copy()
    least_confident_correct = correct_preds.nsmallest(10, 'Max_Probability')
    
    print("\n✅ TOP 10 LEAST CONFIDENT CORRECT PREDICTIONS:")
    print("-" * 60)
    for idx, row in least_confident_correct.iterrows():
        print(f"ID: {row['INC_Number']}")
        print(f"  Class: {row['Real_Label']}")
        print(f"  Confidence: {row['Max_Probability']:.4f}")
        print(f"  Entropy: {row['Entropy']:.4f}")
        print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Error analysis by confidence quartiles
    conf_quartiles = pd.qcut(df_with_conf['Max_Probability'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    error_by_quartile = df_with_conf.groupby(conf_quartiles)['Is_Correct'].agg(['mean', 'count'])
    
    axes[0,0].bar(error_by_quartile.index, 1 - error_by_quartile['mean'], alpha=0.7)
    axes[0,0].set_xlabel('Confidence Quartiles')
    axes[0,0].set_ylabel('Error Rate')
    axes[0,0].set_title('Error Rate by Confidence Quartiles', fontweight='bold')
    
    # Add count annotations
    for i, (quartile, row) in enumerate(error_by_quartile.iterrows()):
        axes[0,0].annotate(f'n={row["count"]}', (i, 1 - row['mean']),
                          ha='center', va='bottom')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Hardest classes to predict (highest error rate)
    class_error_rates = []
    for class_name in class_names:
        class_mask = df_with_conf['Real_Label'] == class_name
        if class_mask.sum() > 0:
            error_rate = (~df_with_conf[class_mask]['Is_Correct']).mean()
            class_error_rates.append({'Class': class_name, 'Error_Rate': error_rate})
    
    error_rates_df = pd.DataFrame(class_error_rates).sort_values('Error_Rate', ascending=True)
    
    axes[0,1].barh(range(len(error_rates_df)), error_rates_df['Error_Rate'], alpha=0.7)
    axes[0,1].set_yticks(range(len(error_rates_df)))
    axes[0,1].set_yticklabels(error_rates_df['Class'])
    axes[0,1].set_xlabel('Error Rate')
    axes[0,1].set_title('Class Difficulty Ranking (Error Rate)', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Confusion vs Confidence scatter
    axes[1,0].scatter(df_with_conf['Max_Probability'], df_with_conf['Entropy'], 
                     c=df_with_conf['Is_Correct'], alpha=0.6, cmap='RdYlGn')
    axes[1,0].set_xlabel('Max Probability (Confidence)')
    axes[1,0].set_ylabel('Entropy (Uncertainty)')
    axes[1,0].set_title('Confidence vs Uncertainty (Green=Correct, Red=Wrong)', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Decision boundary analysis
    # Show probability gap vs correctness
    axes[1,1].boxplot([df_with_conf[df_with_conf['Is_Correct']]['Prob_Gap'],
                      df_with_conf[~df_with_conf['Is_Correct']]['Prob_Gap']],
                     labels=['Correct', 'Incorrect'])
    axes[1,1].set_ylabel('Probability Gap (Top1 - Top2)')
    axes[1,1].set_title('Decision Margin: Correct vs Incorrect Predictions', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return top_confident_wrong, least_confident_correct, error_rates_df

top_confident_wrong, least_confident_correct, error_rates_df = misclassification_deep_dive()

# ============================================================================
# 9. RECOMMENDATIONS AND ACTION ITEMS
# ============================================================================

def generate_recommendations():
    """Generate actionable recommendations based on error analysis"""
    
    print("\n🎯 RECOMMENDATIONS AND ACTION ITEMS")
    print("=" * 70)
    
    # Analyze key issues
    overall_accuracy = (y_true == y_pred).mean()
    
    print(f"\n📊 OVERALL MODEL PERFORMANCE")
    print(f"   • Overall Accuracy: {overall_accuracy:.1%}")
    print(f"   • Macro F1-Score: {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"   • Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.3f}")
    
    # Identify problematic classes
    class_f1_scores = []
    for class_name in class_names:
        if class_name in y_true.values and class_name in y_pred.values:
            f1 = f1_score(y_true, y_pred, labels=[class_name], average=None)[0]
        else:
            f1 = 0
        class_f1_scores.append((class_name, f1))
    
    class_f1_scores.sort(key=lambda x: x[1])
    worst_classes = class_f1_scores[:3]
    best_classes = class_f1_scores[-3:]
    
    print(f"\n🔴 CLASSES NEEDING ATTENTION (Lowest F1-Scores):")
    for class_name, f1 in worst_classes:
        print(f"   • {class_name}: F1 = {f1:.3f}")
    
    print(f"\n🟢 BEST PERFORMING CLASSES:")
    for class_name, f1 in best_classes:
        print(f"   • {class_name}: F1 = {f1:.3f}")
    
    # Confidence analysis insights
    avg_conf_correct = df_with_conf[df_with_conf['Is_Correct']]['Max_Probability'].mean()
    avg_conf_incorrect = df_with_conf[~df_with_conf['Is_Correct']]['Max_Probability'].mean()
    
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"   • Average confidence (correct): {avg_conf_correct:.3f}")
    print(f"   • Average confidence (incorrect): {avg_conf_incorrect:.3f}")
    print(f"   • Confidence gap: {avg_conf_correct - avg_conf_incorrect:.3f}")
    
    # Generate specific recommendations
    recommendations = []
    
    # 1. Data-related recommendations
    if overall_accuracy < 0.8:
        recommendations.append("🔍 DATA QUALITY: Consider collecting more training data, especially for underperforming classes")
    
    # 2. Class imbalance recommendations
    class_counts = y_true.value_counts()
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 3:
        recommendations.append(f"⚖️ CLASS IMBALANCE: Address class imbalance (ratio: {imbalance_ratio:.1f}:1) using techniques like SMOTE, class weights, or focal loss")
    
    # 3. Model confidence recommendations
    if avg_conf_incorrect > 0.7:
        recommendations.append("🎯 OVERCONFIDENCE: Model is overconfident in wrong predictions. Consider calibration techniques")
    
    # 4. Confusion-specific recommendations
    top_confusion = error_pairs.head(3)
    if len(top_confusion) > 0:
        most_confused = top_confusion.iloc[0]
        recommendations.append(f"🔄 CONFUSION PATTERN: Address confusion between '{most_confused['Real_Label']}' and '{most_confused['Predicted_Label']}' ({most_confused['Count']} cases)")
    
    # 5. Feature engineering recommendations
    high_entropy_errors = len(df_with_conf[(~df_with_conf['Is_Correct']) & (df_with_conf['Entropy'] > 2.5)])
    if high_entropy_errors > len(df_with_conf) * 0.1:
        recommendations.append("🔧 FEATURE ENGINEERING: High uncertainty in predictions suggests need for better features")
    
    print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Priority actions
    print(f"\n🚨 PRIORITY ACTIONS:")
    print(f"   1. Focus on improving: {', '.join([c[0] for c in worst_classes])}")
    print(f"   2. Investigate top confusion patterns")
    print(f"   3. Implement confidence calibration")
    print(f"   4. Consider ensemble methods for difficult cases")
    
    return recommendations

recommendations = generate_recommendations()

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================

def create_summary_report():
    """Create a comprehensive summary report"""
    
    print("\n📋 COMPREHENSIVE ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Key metrics
    overall_accuracy = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n🔢 KEY PERFORMANCE METRICS:")
    print(f"   • Overall Accuracy: {overall_accuracy:.1%}")
    print(f"   • Macro F1-Score: {macro_f1:.3f}")
    print(f"   • Weighted F1-Score: {weighted_f1:.3f}")
    print(f"   • Matthews Correlation Coefficient: {matthews_corrcoef(y_true, y_pred):.3f}")
    print(f"   • Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.3f}")
    
    # Error statistics
    total_errors = (~df_with_conf['Is_Correct']).sum()
    error_rate = total_errors / len(df_with_conf)
    
    print(f"\n❌ ERROR STATISTICS:")
    print(f"   • Total Errors: {total_errors:,} out of {len(df_with_conf):,}")
    print(f"   • Error Rate: {error_rate:.1%}")
    print(f"   • Most Common Error: {error_pairs.iloc[0]['Real_Label']} → {error_pairs.iloc[0]['Predicted_Label']} ({error_pairs.iloc[0]['Count']} cases)")
    
    # Confidence insights
    print(f"\n🎯 CONFIDENCE INSIGHTS:")
    print(f"   • Average confidence (correct): {df_with_conf[df_with_conf['Is_Correct']]['Max_Probability'].mean():.3f}")
    print(f"   • Average confidence (incorrect): {df_with_conf[~df_with_conf['Is_Correct']]['Max_Probability'].mean():.3f}")
    print(f"   • Overconfident wrong predictions: {len(df_with_conf[(~df_with_conf['Is_Correct']) & (df_with_conf['Max_Probability'] > 0.8)])}")
    
    # Class-specific insights
    print(f"\n📊 CLASS-SPECIFIC INSIGHTS:")
    for idx, row in error_rates_df.head(3).iterrows():
        print(f"   • Hardest class: {row['Class']} (Error rate: {row['Error_Rate']:.1%})")
    
    print(f"\n✅ ANALYSIS COMPLETED!")
    print(f"   • Analyzed {len(df_with_conf):,} predictions across {len(class_names)} classes")
    print(f"   • Generated {len(recommendations)} specific recommendations")
    print(f"   • Created comprehensive visualizations for error patterns")

create_summary_report()

print("\n" + "="*70)
print("🎉 ERROR ANALYSIS COMPLETE!")
print("="*70)
print("\nTo run this analysis with your data:")
print("1. Replace the sample data section with your CSV loading code")
print("2. Ensure your CSV has the required columns:")
print("   - INC_Number (or similar ID)")
print("   - Real_Label")
print("   - Predicted_Label") 
print("   - Prob_[class_name] for each class")
print("3. Update class_names list with your actual class names")
print("4. Run each section sequentially for comprehensive analysis")
print("\nThe analysis provides:")
print("• Confusion matrices and performance metrics")
print("• Class-wise error analysis")
print("• Confidence and calibration analysis")
print("• ROC and Precision-Recall curves")
print("• Probability distribution analysis")
print("• Detailed misclassification investigation")
print("• Actionable recommendations")
