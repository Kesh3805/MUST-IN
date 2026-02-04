"""
Download and verify transformer models for MUST-IN framework.

Usage:
    python scripts/download_transformers.py --all
    python scripts/download_transformers.py --model bert-base-multilingual-cased
    python scripts/download_transformers.py --model xlm-roberta-base --verify
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Available models
MODELS = {
    "mbert-cased": "bert-base-multilingual-cased",
    "mbert-uncased": "bert-base-multilingual-uncased",
    "xlm-roberta": "xlm-roberta-base",
    "xlm-roberta-large": "xlm-roberta-large",
    "indic-bert": "ai4bharat/indic-bert",
    "muril": "google/muril-base-cased",
}

def download_model(model_name: str, num_labels: int = 3) -> bool:
    """
    Download a transformer model and tokenizer.
    
    Args:
        model_name: Model identifier (e.g., "bert-base-multilingual-cased")
        num_labels: Number of classification labels (default: 3)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Downloading: {model_name}")
        print(f"{'='*60}")
        
        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer downloaded")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        
        # Download model
        print("📥 Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        print(f"✓ Model downloaded")
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        print(f"  Size: ~{num_params * 4 / (1024**2):.1f} MB")
        
        # Test model
        print("🧪 Testing model...")
        test_text = "This is a test sentence."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        print(f"✓ Model test successful")
        print(f"  Output shape: {logits.shape}")
        print(f"  Predictions: {probs[0].tolist()}")
        
        print(f"\n✅ Successfully downloaded: {model_name}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {model_name}: {e}\n")
        return False


def verify_model(model_name: str) -> bool:
    """
    Verify that a model is already downloaded and working.
    
    Args:
        model_name: Model identifier
    
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        print(f"Verifying: {model_name}...", end=" ")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download transformer models for MUST-IN"
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Model to download (can be specified multiple times)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all recommended models"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    # List available models
    if args.list:
        print("\n📋 Available Models:")
        print(f"{'='*60}")
        for alias, name in MODELS.items():
            print(f"  {alias:<20} → {name}")
        print(f"{'='*60}\n")
        print("Usage examples:")
        print("  python scripts/download_transformers.py --model mbert-cased")
        print("  python scripts/download_transformers.py --all")
        print("  python scripts/download_transformers.py --verify")
        return
    
    # Verify mode
    if args.verify:
        print("\n🔍 Verifying Models:")
        print(f"{'='*60}")
        
        models_to_verify = []
        if args.all:
            models_to_verify = list(MODELS.values())
        elif args.model:
            models_to_verify = [MODELS.get(m, m) for m in args.model]
        else:
            # Verify default models
            models_to_verify = [
                "bert-base-multilingual-cased",
                "xlm-roberta-base"
            ]
        
        verified = 0
        total = len(models_to_verify)
        
        for model_name in models_to_verify:
            if verify_model(model_name):
                verified += 1
        
        print(f"{'='*60}")
        print(f"✓ Verified: {verified}/{total} models")
        
        if verified < total:
            print(f"⚠ Missing: {total - verified} models")
            print("Run without --verify to download missing models")
        
        return
    
    # Download mode
    print("\n🚀 MUST-IN Transformer Model Downloader")
    print(f"{'='*60}")
    
    # Check PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")
    
    # Determine which models to download
    models_to_download = []
    
    if args.all:
        print("\n📦 Downloading all recommended models...")
        models_to_download = [
            "bert-base-multilingual-cased",
            "bert-base-multilingual-uncased",
            "xlm-roberta-base"
        ]
    elif args.model:
        # Resolve aliases
        for model in args.model:
            if model in MODELS:
                models_to_download.append(MODELS[model])
            else:
                models_to_download.append(model)
    else:
        # Default: download mBERT cased
        print("\n📦 Downloading default model (mBERT-cased)...")
        print("Use --all to download all models or --list to see options")
        models_to_download = ["bert-base-multilingual-cased"]
    
    # Download models
    successful = 0
    failed = 0
    
    for model_name in models_to_download:
        if download_model(model_name):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Download Summary:")
    print(f"{'='*60}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"{'='*60}\n")
    
    if successful > 0:
        print("✅ Models ready for training!")
        print("\nNext steps:")
        print("  1. Verify models: python scripts/download_transformers.py --verify")
        print("  2. Start training: python main.py --run-dl --save-models")
        print("  3. See TRAINING_WORKFLOW.md for complete guide")
    
    if failed > 0:
        print("\n⚠ Some models failed to download.")
        print("  - Check internet connection")
        print("  - Check disk space (need ~2GB per model)")
        print("  - Try downloading individual models")


if __name__ == "__main__":
    main()
