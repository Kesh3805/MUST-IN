"""
MUST++ Metrics and Evaluation Module

Produces:
- Confusion matrix
- Per-label precision, recall, F1
- False Negative Rate (FNR) for hate
- Fallback activation rates
- Pipeline coverage statistics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import json


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for MUST++ pipeline."""
    
    # Per-label metrics
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    support: Dict[str, int]
    
    # Confusion matrix
    confusion_matrix: np.ndarray
    labels: List[str]
    
    # Critical safety metrics
    hate_fnr: float  # False Negative Rate for hate (CRITICAL)
    hate_fpr: float  # False Positive Rate for hate
    
    # Pipeline statistics
    fallback_rate: float
    escalation_rate: float
    degraded_mode_rate: float
    
    # Coverage statistics
    avg_confidence: float
    avg_entropy: float
    avg_tokenization_coverage: float
    
    # Per-language breakdown
    language_distribution: Dict[str, int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "support": self.support,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "labels": self.labels,
            "hate_fnr": self.hate_fnr,
            "hate_fpr": self.hate_fpr,
            "fallback_rate": self.fallback_rate,
            "escalation_rate": self.escalation_rate,
            "degraded_mode_rate": self.degraded_mode_rate,
            "avg_confidence": self.avg_confidence,
            "avg_entropy": self.avg_entropy,
            "avg_tokenization_coverage": self.avg_tokenization_coverage,
            "language_distribution": self.language_distribution
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "MUST++ EVALUATION SUMMARY",
            "=" * 60,
            "",
            "PER-LABEL METRICS:",
            "-" * 40,
        ]
        
        for label in self.labels:
            lines.append(
                f"  {label:12} | P: {self.precision.get(label, 0):.3f} | "
                f"R: {self.recall.get(label, 0):.3f} | "
                f"F1: {self.f1_score.get(label, 0):.3f} | "
                f"N: {self.support.get(label, 0)}"
            )
        
        lines.extend([
            "",
            "CRITICAL SAFETY METRICS:",
            "-" * 40,
            f"  Hate False Negative Rate (FNR): {self.hate_fnr:.4f}",
            f"  Hate False Positive Rate (FPR): {self.hate_fpr:.4f}",
            "",
            "PIPELINE STATISTICS:",
            "-" * 40,
            f"  Fallback Rate: {self.fallback_rate:.2%}",
            f"  Escalation Rate: {self.escalation_rate:.2%}",
            f"  Degraded Mode Rate: {self.degraded_mode_rate:.2%}",
            "",
            "COVERAGE STATISTICS:",
            "-" * 40,
            f"  Avg Confidence: {self.avg_confidence:.3f}",
            f"  Avg Entropy: {self.avg_entropy:.3f}",
            f"  Avg Tokenization Coverage: {self.avg_tokenization_coverage:.3f}",
            "",
            "CONFUSION MATRIX:",
            "-" * 40,
        ])
        
        # Add confusion matrix
        header = "         " + "".join(f"{l:>12}" for l in self.labels)
        lines.append(header)
        for i, label in enumerate(self.labels):
            row = f"{label:>8} " + "".join(f"{self.confusion_matrix[i, j]:>12}" for j in range(len(self.labels)))
            lines.append(row)
        
        lines.extend([
            "",
            "LANGUAGE DISTRIBUTION:",
            "-" * 40,
        ])
        for lang, count in sorted(self.language_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"  {lang}: {count}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class MUSTPlusEvaluator:
    """
    Evaluator for MUST++ pipeline.
    
    Computes all required metrics from predictions and ground truth.
    """
    
    LABELS = ["neutral", "offensive", "hate"]
    LABEL_TO_IDX = {"neutral": 0, "offensive": 1, "hate": 2}
    
    def __init__(self):
        """Initialize evaluator."""
        self.predictions = []
        self.ground_truth = []
        self.outputs = []
    
    def add_sample(
        self, 
        prediction: str, 
        ground_truth: str, 
        output: Optional[Dict] = None
    ):
        """
        Add a sample for evaluation.
        
        Args:
            prediction: Predicted label
            ground_truth: Ground truth label
            output: Optional full MUSTPlusOutput as dict
        """
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        if output:
            self.outputs.append(output)
    
    def evaluate(self) -> EvaluationMetrics:
        """
        Compute all evaluation metrics.
        
        Returns:
            EvaluationMetrics with complete results
        """
        n = len(self.predictions)
        if n == 0:
            raise ValueError("No samples to evaluate")
        
        # Build confusion matrix
        confusion = np.zeros((3, 3), dtype=int)
        for pred, truth in zip(self.predictions, self.ground_truth):
            pred_idx = self.LABEL_TO_IDX.get(pred, 0)
            truth_idx = self.LABEL_TO_IDX.get(truth, 0)
            confusion[truth_idx, pred_idx] += 1
        
        # Per-label metrics
        precision = {}
        recall = {}
        f1_score = {}
        support = {}
        
        for label in self.LABELS:
            idx = self.LABEL_TO_IDX[label]
            tp = confusion[idx, idx]
            fp = confusion[:, idx].sum() - tp
            fn = confusion[idx, :].sum() - tp
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            precision[label] = p
            recall[label] = r
            f1_score[label] = f1
            support[label] = int(confusion[idx, :].sum())
        
        # Critical: Hate FNR (false negatives for hate are CRITICAL)
        hate_idx = self.LABEL_TO_IDX["hate"]
        hate_fn = confusion[hate_idx, :].sum() - confusion[hate_idx, hate_idx]
        hate_tp = confusion[hate_idx, hate_idx]
        hate_tn = confusion.sum() - confusion[hate_idx, :].sum() - confusion[:, hate_idx].sum() + hate_tp
        hate_fp = confusion[:, hate_idx].sum() - hate_tp
        
        hate_fnr = hate_fn / (hate_fn + hate_tp) if (hate_fn + hate_tp) > 0 else 0.0
        hate_fpr = hate_fp / (hate_fp + hate_tn) if (hate_fp + hate_tn) > 0 else 0.0
        
        # Pipeline statistics from outputs
        fallback_count = 0
        escalation_count = 0
        degraded_count = 0
        total_confidence = 0.0
        total_entropy = 0.0
        total_coverage = 0.0
        language_counts = Counter()
        
        for output in self.outputs:
            if output.get("fallback_used", False):
                fallback_count += 1
            if output.get("escalation_triggered", False):
                escalation_count += 1
            if output.get("degraded_mode", False):
                degraded_count += 1
            
            total_confidence += output.get("confidence", 0.0)
            total_entropy += output.get("entropy", 0.0)
            total_coverage += output.get("tokenization_coverage", 0.0)
            
            # Count languages
            langs = output.get("languages_detected", {})
            for lang in langs:
                language_counts[lang] += 1
        
        n_outputs = len(self.outputs) or 1
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            support=support,
            confusion_matrix=confusion,
            labels=self.LABELS,
            hate_fnr=hate_fnr,
            hate_fpr=hate_fpr,
            fallback_rate=fallback_count / n_outputs,
            escalation_rate=escalation_count / n_outputs,
            degraded_mode_rate=degraded_count / n_outputs,
            avg_confidence=total_confidence / n_outputs,
            avg_entropy=total_entropy / n_outputs,
            avg_tokenization_coverage=total_coverage / n_outputs,
            language_distribution=dict(language_counts)
        )
    
    def clear(self):
        """Clear all samples."""
        self.predictions = []
        self.ground_truth = []
        self.outputs = []


def evaluate_pipeline_on_dataset(
    pipeline,
    texts: List[str],
    labels: List[str]
) -> EvaluationMetrics:
    """
    Evaluate pipeline on a labeled dataset.
    
    Args:
        pipeline: MUSTPlus pipeline instance
        texts: List of input texts
        labels: List of ground truth labels
        
    Returns:
        EvaluationMetrics
    """
    evaluator = MUSTPlusEvaluator()
    
    for text, label in zip(texts, labels):
        output = pipeline.classify(text)
        evaluator.add_sample(
            prediction=output.label,
            ground_truth=label,
            output=output.to_dict()
        )
    
    return evaluator.evaluate()


def document_blind_spots(metrics: EvaluationMetrics) -> Dict[str, str]:
    """
    Document known blind spots based on metrics.
    
    Returns dict of issue -> description.
    """
    blind_spots = {}
    
    # High FNR for hate is critical
    if metrics.hate_fnr > 0.1:
        blind_spots["high_hate_fnr"] = (
            f"Hate FNR is {metrics.hate_fnr:.2%} - above 10% threshold. "
            "Pipeline may miss hate speech. Review lexicon coverage and fallback logic."
        )
    
    # Low recall for any label
    for label in ["neutral", "offensive", "hate"]:
        if metrics.recall.get(label, 0) < 0.7:
            blind_spots[f"low_{label}_recall"] = (
                f"{label} recall is {metrics.recall[label]:.2%} - below 70% threshold. "
                "May be under-classifying this label."
            )
    
    # High fallback rate may indicate model issues
    if metrics.fallback_rate > 0.5:
        blind_spots["high_fallback"] = (
            f"Fallback rate is {metrics.fallback_rate:.2%} - over 50%. "
            "Primary model may be underconfident or undertrained."
        )
    
    # Low tokenization coverage
    if metrics.avg_tokenization_coverage < 0.8:
        blind_spots["low_tokenization"] = (
            f"Avg tokenization coverage is {metrics.avg_tokenization_coverage:.2%}. "
            "Tokenizer may not handle Romanized/code-mixed text well."
        )
    
    # High entropy
    if metrics.avg_entropy > 1.0:
        blind_spots["high_entropy"] = (
            f"Avg entropy is {metrics.avg_entropy:.2f} - model predictions are uncertain. "
            "May need more training data or better calibration."
        )
    
    return blind_spots


# =========================================
# DEPLOYMENT READINESS CHECKLIST
# =========================================

def generate_readiness_checklist(metrics: EvaluationMetrics) -> Dict[str, bool]:
    """
    Generate deployment readiness checklist.
    
    Returns dict of requirement -> passed (True/False).
    """
    checklist = {}
    
    # Critical safety requirements
    checklist["hate_fnr_below_5%"] = metrics.hate_fnr < 0.05
    checklist["hate_recall_above_90%"] = metrics.recall.get("hate", 0) > 0.90
    
    # Quality requirements
    checklist["overall_f1_above_80%"] = (
        sum(metrics.f1_score.values()) / len(metrics.f1_score) > 0.80
    )
    checklist["neutral_precision_above_85%"] = metrics.precision.get("neutral", 0) > 0.85
    
    # Pipeline health
    checklist["fallback_rate_below_30%"] = metrics.fallback_rate < 0.30
    checklist["tokenization_coverage_above_80%"] = metrics.avg_tokenization_coverage > 0.80
    
    # Consistency
    checklist["degraded_mode_rate_below_5%"] = metrics.degraded_mode_rate < 0.05
    
    return checklist


def print_readiness_report(
    metrics: EvaluationMetrics,
    checklist: Dict[str, bool],
    blind_spots: Dict[str, str]
):
    """Print deployment readiness report."""
    print("\n" + "=" * 70)
    print("MUST++ DEPLOYMENT READINESS REPORT")
    print("=" * 70)
    
    # Metrics summary
    print(metrics.summary())
    
    # Checklist
    print("\nDEPLOYMENT READINESS CHECKLIST:")
    print("-" * 50)
    all_passed = True
    for requirement, passed in checklist.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {requirement}: {status}")
        if not passed:
            all_passed = False
    
    # Blind spots
    if blind_spots:
        print("\nDOCUMENTED BLIND SPOTS:")
        print("-" * 50)
        for issue, description in blind_spots.items():
            print(f"  ⚠️ {issue}:")
            print(f"     {description}")
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_passed and not blind_spots:
        print("✅ PIPELINE IS DEPLOYMENT READY")
    elif all_passed:
        print("⚠️ PIPELINE PASSES CHECKS BUT HAS BLIND SPOTS - REVIEW RECOMMENDED")
    else:
        print("❌ PIPELINE IS NOT DEPLOYMENT READY - ADDRESS FAILURES")
    print("=" * 70)
