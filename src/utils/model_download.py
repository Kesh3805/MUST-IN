"""Preload pretrained transformer models used in the paper."""
from typing import Iterable, List

from transformers import AutoModelForSequenceClassification, AutoTokenizer


PAPER_MODELS: List[str] = [
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
]


def preload_models(model_names: Iterable[str] = None) -> List[str]:
    """
    Download tokenizer + sequence classification head for each model.

    Returns:
        List of successfully downloaded model names.
    """
    if model_names is None:
        model_names = PAPER_MODELS

    downloaded = []
    for name in model_names:
        try:
            AutoTokenizer.from_pretrained(name)
            AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)
            downloaded.append(name)
        except Exception:
            continue

    return downloaded
