from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Evaluator:
    """
    Implements Section 6: Evaluation
    """
    
    def __init__(self, labels):
        self.labels = labels

    def evaluate(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """
        Compute all metrics
        """
        print(f"--- Evaluation Results for {model_name} ---")
        
        # 6.1 Basic Metrics
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.labels))
        
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        # 6.2 Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, model_name)
        
        # ROC-AUC (One-vs-Rest)
        # Needs y_prob (probabilities)
        if y_prob is not None:
             # Handle multiclass ROC-AUC
             try:
                # Assuming y_true matches classes 0,1,2
                roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                print(f"ROC-AUC (OvR): {roc:.4f}")
             except Exception as e:
                 print(f"Could not compute ROC-AUC: {e}")

    def plot_confusion_matrix(self, cm, title):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.labels, yticklabels=self.labels, cmap='Blues')
        plt.title(f'Confusion Matrix: {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        # We won't show() here because it blocks execution in scripts, usually save is better
        plt.savefig(f'results/{title}_confusion_matrix.png')
        print(f"Confusion matrix saved to results/{title}_confusion_matrix.png")
        plt.close()

